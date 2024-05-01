from idaes.core.solvers import get_solver
import reaktoro as rkt
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
import numpy as np
from scipy.sparse import coo_matrix
import math

class ReaktoroOutputModel(ExternalGreyBoxModel):
########################################################################################
    # custom Grey Box functions
    def get_rkt_state(self, state):
        # assign a Reaktoro state object to instance
        self.state = state

    def get_pyo_model(self, model, inputs, outputs):
        # assign a Pyomo model object to instance
        self.model = model
        # get inputs and outputs from Pyomo model
        self.inputs = inputs
        self.outputs = outputs

########################################################################################
    # standard Grey Box functions
    def input_names(self):
        # get input names (required by Grey Box)
        return self.inputs.keys()

    def output_names(self):
        # get output names (not required, but helpful)
        return self.outputs

    def set_input_values(self, input_values):
        # set input values from Pyomo as inputs to External Model (required by Grey Box)
        self._input_values = list(input_values)

    def finalize_block_construction(self, pyomo_block):
        # initialize Pyomo block for External Model
        block_components = [obj for obj in pyomo_block.component_objects(pyo.Var)]
        for block in block_components:
            # 1e-16 is Reaktoro's epsilon value
            if block.name == "reactor.inputs":
                for var in block:
                    block[var].setlb(1e-16)
                    # self.inputs is a dictionary with names and initial values
                    block[var].value = pyo.value(self.inputs[var])
            elif block.name == "reactor.outputs":
                for var in block:
                    block[var].setlb(1e-16)
                    # set output amounts (mol) to the initial_species amounts set in Pyomo model
                    block[var].value = pyo.value(.1)

    def evaluate_outputs(self):
        # update Reaktoro state with current inputs (this function runs repeatedly)
        params = dict(zip(self.inputs.keys(), self._input_values))
        # get Jacobian of species amounts (mol) w.r.t. input vars
        self.jacobian_matrix = solve_eq_problem(self.state, params=params)
        self.model.pH = rkt.AqueousProps(self.state.props()).pH().val()
        result = np.array([self.state.speciesAmount(output).val() for output in self.outputs], dtype=np.float64)
        return result

    def evaluate_jacobian_outputs(self):
        # get coordinate matrix from Jacobian matrix
        self.evaluate_outputs()
        jm = self.jacobian_matrix
        i = np.array([i for i in range(jm.shape[0]) for j in range(jm.shape[1])])
        j = np.array([j for i in range(jm.shape[0]) for j in range(jm.shape[1])])
        cm = coo_matrix((jm.flatten(), (i, j)))
        return cm

########################################################################################
# model building functions

def build_pyo_model(state, inputs, outputs):
    # create model
    model = pyo.ConcreteModel()
    model.name = "Reaktoro-Pyomo"
    model.species = pyo.Set(initialize=outputs)
    # specify the initial species amounts in Pyomo model
    model.initial_species = pyo.Var(
        model.species, domain=pyo.NonNegativeReals, initialize=1e-16, units=pyo.units.dimensionless,
    )
    model.equilibrium_species = pyo.Var(
        model.species, domain=pyo.NonNegativeReals, initialize=1e-16, units=pyo.units.dimensionless,
    )
    for species in model.initial_species:
        model.initial_species[species].fix(state.props().speciesActivity(species).val())
        model.equilibrium_species[species].fix(state.props().speciesActivity(species).val())

    # create external model
    external_model = ReaktoroOutputModel()
    external_model.get_pyo_model(model, inputs=inputs, outputs=outputs)
    external_model.get_rkt_state(state)
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

properties = {
    # system props
    "total": [
        "temperature",
        "pressure",
        "volume",
        "enthalpy",
        "entropy",
        "internalEnergy",
        "helmholtzEnergy",
        "charge",
    ],
    # iterate thru elements
    "element": [
        "elementAmount",
    ],
    # iterate thru species
    "species": [
        "speciesAmount",
        "speciesMoleFraction",
        "speciesActivityCoefficient",
        "speciesActivity",
        "speciesActivityLg",
        "speciesActivityLn",
        "speciesChemicalPotential",
        "speciesStandardVolume",
        "speciesStandardGibbsEnergy",
        "speciesStandardEnthalpy",
        "speciesStandardEntropy",
        "speciesStandardInternalEnergy",
        "speciesStandardHelmholtzEnergy",
        "speciesStandardHeatCapacityConstP",
        "speciesStandardHeatCapacityConstV",
    ],
}

def _get_chem_props(state, properties=properties):
    # compile list of chem props for Jacobian (needs to be tested with dudw function)
    chem_props = {}
    for key, props in properties.items():
        if key == "total":
            for prop in props:
                chem_props[f"{prop}_total"] = getattr(state.props(), prop)().val()
        elif key == "element":
            for prop in props:
                for element in state.system().elements():
                    symbol = element.symbol()
                    chem_props[f"{prop}_{symbol}"] = getattr(state.props(), prop)(symbol).val()
        elif key == "species":
            for prop in props:
                for species in state.system().species():
                    name = species.name()
                    chem_props[f"{prop}_{name}"] = getattr(state.props(), prop)(name).val()
    return chem_props

valid_kws = ["temperature", "pH", "pressure"]
def _add_spec(specs, param):
    # add a specification to Reaktoro equilibrium problem
    specs.openTo(param)
    idx = specs.addInput(param)
    constraint = rkt.EquationConstraint()
    constraint.id = f"{param}_constraint"
    if "_dose" in param:
        param = param.split("_")[0]
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

def _set_eq_conds(specs, params=None, valid_kws=valid_kws):
    # set conditions for Reaktoro equilibrium problem (values that must be met)
    conditions = rkt.EquilibriumConditions(specs)
    if params is not None:
        for param, val in params.items():
            try:
                if param in valid_kws:
                    getattr(conditions, param)(val)
                else:
                    conditions.set(param, val)
            except:
                print(f"Failed to add param to equilibrium conditions: {param}.")
    return conditions

def _get_jacobian_matrix(state, conditions, sensitivity):
    # get Jacobian matrix from Reaktoro given species list (n) and input variables (w)
    jacobian_columns = conditions.inputNames()
    jacobian_matrix = sensitivity.dndw()
    jacobian_rows = [species.name() for species in state.system().species()]
    assert jacobian_matrix.shape == (len(jacobian_rows), len(jacobian_columns))
    return jacobian_matrix

def solve_eq_problem(state, params=None):
    # solve Reaktoro equilibrium problem using the above helper functions
    if params is not None:
        specs = _set_eq_specs(state.system(), params)
        solver = rkt.EquilibriumSolver(specs)
        sensitivity = rkt.EquilibriumSensitivity(specs)
        conditions = _set_eq_conds(specs, params)
        result = solver.solve(state, sensitivity, conditions)
        jacobian_matrix = _get_jacobian_matrix(state, conditions, sensitivity)
    else:
        solver = rkt.EquilibriumSolver(state.system())
        result = solver.solve(state)
        jacobian_matrix = None
    if not result.succeeded():
        print("Equilibrium calculation failed to converge ... ")
    state.props().update(state)
    return jacobian_matrix
