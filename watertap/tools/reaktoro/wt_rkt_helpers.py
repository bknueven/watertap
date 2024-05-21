import reaktoro as rkt
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
import numpy as np
from scipy.sparse import coo_matrix


class ReaktoroOutputModel(ExternalGreyBoxModel):
########################################################################################
    # custom Grey Box functions
    def get_rkt_state(self, state, rkt_input, rkt_output):
        # assign a Reaktoro state object to instance
        self.state = state
        self.inputs = rkt_input
        self.output_property = rkt_output
        self.outputs = _get_jacobian_rows(state, rkt_output)

########################################################################################
    # standard Grey Box functions
    def input_names(self):
        # get input names (required by Grey Box)
        return self.inputs.keys()

    def output_names(self):
        # get output names (not required, but helpful)
        return [species for idx, (prop, species) in self.outputs]

    def set_input_values(self, input_values):
        # set input values from Pyomo as inputs to External Model (required by Grey Box)
        self._input_values = list(input_values)

    def finalize_block_construction(self, pyomo_block):
        # initialize Pyomo block for External Model
        block_components = [obj for obj in pyomo_block.component_objects(pyo.Var)]
        for block in block_components:
            # 1e-16 is Reaktoro's epsilon value
            if block.name == "reactor.inputs":
                for var, val in self.inputs.items():
                    block[var].setlb(0)
                    block[var].value = val
            elif block.name == "reactor.outputs":
                for idx, (prop, var) in self.outputs:
                    block[var].setlb(0)
                    block[var].value = getattr(self.state.props(), prop)(var).val()

    def evaluate_outputs(self):
        # update Reaktoro state with current inputs (this function runs repeatedly)
        params = dict(zip(self.inputs.keys(), self._input_values))
        # get Jacobian of species amounts (mol) w.r.t. input vars
        self.jacobian_matrix = solve_eq_problem(self.state, params=params)
        result = []
        for idx, (prop, species) in self.outputs:
            result.append(getattr(self.state.props(), prop)(species).val())
        return np.array(result, dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        # get coordinate matrix from Jacobian matrix
        self.evaluate_outputs()
        jm = self.jacobian_matrix[[idx for idx, (prop, species) in self.outputs]]
        i = np.array([i for i in range(jm.shape[0]) for j in range(jm.shape[1])])
        j = np.array([j for i in range(jm.shape[0]) for j in range(jm.shape[1])])
        cm = coo_matrix((jm.flatten(), (i, j)))
        return cm

########################################################################################
# model building functions

def build_pyo_model(state, rkt_input, rkt_output):
    # create model
    model = pyo.ConcreteModel()
    model.name = "Reaktoro-Pyomo"
    # create external model
    external_model = ReaktoroOutputModel()
    external_model.get_rkt_state(state, rkt_input, rkt_output)
    model.reactor = ExternalGreyBoxBlock(external_model=external_model)
    return model

def build_rkt_state(db, aq_species, gas_species, solid_species, temp, pres, allow_solids):
    # create Reaktoro model phases
    aq_phase = rkt.AqueousPhase(rkt.speciate([species for species in aq_species]))
    aq_phase.set(rkt.ActivityModelPitzer())
    phases = [aq_phase]
    if gas_species:
        gas_phase = rkt.GaseousPhase([species for species in gas_species])
        phases.append(gas_phase)
    else:
        gas_phase = None
    if allow_solids:
        if solid_species:
            phases.extend([rkt.MineralPhase(species) for species in solid_species])
        else:
            phases.append(rkt.MineralPhases())
    system = rkt.ChemicalSystem(db, *phases)
    state = rkt.ChemicalState(system)
    # set temperature and pressure (add other state vars later)
    state.temperature(pyo.value(temp), str(pyo.units.get_units(temp)))
    state.pressure(pyo.value(pres), str(pyo.units.get_units(pres)))
    # set initial values from input species
    input_species = {}
    for phase in [aq_species, gas_species, solid_species]:
        if phase is not None:
            input_species.update({**phase})
    for species, conc in input_species.items():
        if pyo.value(conc) != 0:
            state.set(species, pyo.value(conc), str(pyo.units.get_units(conc)))
    state.props().update(state)
    return state

########################################################################################
# solver functions for Reaktoro equilibrium problems

valid_kws = ["temperature", "pH", "pressure"]
def _add_spec(specs, param):
    # add a specification to Reaktoro equilibrium problem
    specs.openTo(param)
    idx = specs.addInput(param)
    constraint = rkt.EquationConstraint()
    constraint.id = f"{param}_constraint"
    constraint.fn = lambda props, w: props.speciesAmount(param) - w[idx]
    specs.addConstraint(constraint)

def _set_eq_specs(system, params=None, valid_kws=valid_kws):
    # set specifications for Reaktoro equilibrium problem (variables involved in calculation)
    specs = rkt.EquilibriumSpecs(system)
    if params is not None:
        for param in params:
            try:
                if param in valid_kws:
                    getattr(specs, param)()
                else:
                    _add_spec(specs, param)
            except:
                print(f"Failed to add param to equilibrium specs: {param}.")
    return specs

def _set_eq_conds(state, specs, params=None, valid_kws=valid_kws):
    # set conditions for Reaktoro equilibrium problem (values that must be met)
    conditions = rkt.EquilibriumConditions(specs)
    if params is not None:
        for param, val in params.items():
            try:
                if param in valid_kws:
                    getattr(conditions, param)(val)
                else:
                    initial_val = state.speciesAmount(param)
                    conditions.set(param, initial_val + val)

            except:
                print(f"Failed to add param to equilibrium conditions: {param}.")
    return conditions

def _get_jacobian_rows(state, prop):
    species = [species.name() for species in state.system().species()]
    phases = [phase.name() for phase in state.system().phases()]
    # state temperature and pressure
    rows = [("temperature",), ("pressure",)]
    # species amounts * # species
    rows.extend([("speciesAmount", s) for s in species])
    # surface area of reacting surfaces (probably will be here if kinetics are used)
    # temperature of each phase * # phases
    rows.extend([("temperature", p) for p in phases])
    # pressure of each phase * # phases
    rows.extend([("pressure", p) for p in phases])
    # sum speciesAmounts * # phases
    rows.extend([("amount", p) for p in phases])
    # sum speciesMasses * # phases
    rows.extend([("mass", p) for p in phases])
    # mole fractions * # species
    rows.extend([("speciesMoleFraction", s) for s in species])
    # standard molar gibbs free energy * # species
    rows.extend([("speciesStandardGibbsEnergy", s) for s in species])
    # standard molar enthalpy * # species
    rows.extend([("speciesStandardEnthalpy", s) for s in species])
    # standard molar volumes * # species
    rows.extend([("speciesStandardVolume", s) for s in species])
    # temp derivative of standard molar volumes * # species
    rows.extend([("speciesStandardVolumeT", s) for s in species])
    # pres derivative of standard molar volumes * # species
    rows.extend([("speciesStandardVolumeP", s) for s in species])
    # standard isobaric molar heat capacities * # species
    rows.extend([("speciesStandardHeatCapacityConstP", s) for s in species])
    # corrective molar volume * # phases
    rows.extend([("molarVolume", p) for p in phases])
    # temp derivative of corrective molar volume * # phases
    rows.extend([("molarVolumeT", p) for p in phases])
    # pres derivative of corrective molar volume * # phases
    rows.extend([("molarVolumeP", p) for p in phases])
    # mole frac derivative of corrective molar volume * # phases
    rows.extend([("molarVolumeI", s) for s in species]) # not sure what this is
    # corrective gibbs free energy * # phases
    rows.extend([("molarGibbsEnergy", p) for p in phases])
    # corrective molar enthalpy * # phases
    rows.extend([("molarEnthalpy", p) for p in phases])
    # corrective isobaric molar heat capacity * # phases
    rows.extend([("molarHeatCapacityConstP", p) for p in phases])
    # activity coefficients (ln) * # species
    rows.extend([("speciesActivityCoefficientLn", s) for s in species])
    # activities (ln) * # species
    rows.extend([("speciesActivity", s) for s in species])
    # chemical potentials * # species
    rows.extend([("speciesChemicalPotential", s) for s in species])
    jacobian_rows = [(idx, row) for idx, row in enumerate(rows) if prop == row[0]]
    return jacobian_rows

def solve_eq_problem(state, params=None):
    # solve Reaktoro equilibrium problem using the above helper functions
    if params is None:
        solver = rkt.EquilibriumSolver(state.system())
        result = solver.solve(state)
        jacobian_matrix = None
    else:
        specs = _set_eq_specs(state.system(), params)
        solver = rkt.EquilibriumSolver(specs)
        sensitivity = rkt.EquilibriumSensitivity(specs)
        conditions = _set_eq_conds(state, specs, params)
        result = solver.solve(state, sensitivity, conditions)
        jacobian_matrix = sensitivity.dudw()
    assert result.succeeded()
    state.props().update(state)
    return jacobian_matrix
