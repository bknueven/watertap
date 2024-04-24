from idaes.core.solvers import get_solver
import reaktoro as rkt
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
import numpy as np

class ReaktoroOutputModel(ExternalGreyBoxModel):

    def get_rkt_state(self, state):
        self.state = state

    def get_model_inputs(self, model):
        self.inputs = [obj.name for obj in model.component_data_objects(pyo.Var)]
        #print(self.inputs)

    def input_names(self):
        return self.inputs

    def output_names(self):
        return self.inputs

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def finalize_block_construction(self, pyomo_block):
        # set bounds and initialize inputs and outputs
        block_components = [obj for obj in pyomo_block.component_objects(pyo.Var)]
        model = pyomo_block.model()
        for block in block_components:
            for var in block:
                block[var].setlb(0)
                if var == "ph":
                    block[var].setub(14)
                    unit = pyo.units.dimensionless
                elif var == "temperature":
                    unit = pyo.units.K
                elif var == "pressure":
                    unit = pyo.units.Pa
                elif "species_amounts" in var:
                    unit = pyo.units.mol
                block[var].value = model.find_component(var) * unit
                # not sure if units are useful or not

    def evaluate_outputs(self):
        # call external solve function
        solve_rkt_state(state, temp, pres)
        # return derivatives from Reaktoro
        ret = update_pyo_model(state, self.model())
        print(ret)
        return np.asarray(ret, dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        y0 = self.evaluate_outputs(state)
        jacobian_matrix = np.array() # apply derivatives from Reaktoro
        return jacobian_matrix

def build_pyo_model(state):
    # create model
    model = pyo.ConcreteModel()
    model.name = "Reaktoro-Pyomo"
    # index system species
    model.all_species = pyo.Set(initialize=(species.name() for species in state.system().species()))
    return model

def update_pyo_model(state, model):
    # external solve function
    aq_props = rkt.AqueousProps(state.props())
    model.ph = aq_props.pH().val()
    model.temperature = state.temperature().val()
    model.pressure = state.pressure().val()
    for item in model.species_amounts:
        model.species_amounts[item] = state.speciesAmount(item).val()
    return

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

def update_rkt_state(state, temp, pres, ph, specs=None, conditions=None):
    if specs is None:
        specs = _setup_eq_problem(state.system(), temp, pres, ph, specs)
    jacobian_matrix = _solve_eq_problem(state, specs, temp, pres, ph, conditions)
    return jacobian_matrix

# the two functions below should remain separate from above function to allow for custom constraints in Reaktoro
def _setup_eq_problem(system, temp, pres, ph, specs=None):
    specs = rkt.EquilibriumSpecs(system)
    if temp is not None:
        specs.temperature()
    if pres is not None:
        specs.pressure()
    if ph is not None:
        specs.pH()
    return specs

def _solve_eq_problem(state, specs, temp, pres, ph, conditions=None):
    # solve and extract derivatives needed for model solver
    solver = rkt.EquilibriumSolver(specs)
    sensitivity = rkt.EquilibriumSensitivity(specs)
    conditions = rkt.EquilibriumConditions(specs)
    if temp is not None:
        conditions.temperature(pyo.value(temp), str(pyo.units.get_units(temp)))
    if pres is not None:
        conditions.pressure(pyo.value(pres), str(pyo.units.get_units(pres)))
    if ph is not None:
        conditions.pH(pyo.value(ph), str(pyo.units.get_units(ph)))
    result = solver.solve(state, sensitivity, conditions)
    assert result.succeeded()
    state.props().update(state)
    jacobian_matrix = sensitivity.dndw()
    print(f"jacobian columns: {state.equilibrium().namesInputVariables()}\n" +
          f"jacobian rows: {[species.name() for species in state.system().species()]}")
    return jacobian_matrix
