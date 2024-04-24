import reaktoro as rkt
import pyomo.environ as pyo
from reaktoro_softening import ReaktoroOutputModel, build_rkt_state, update_rkt_state, build_pyo_model, update_pyo_model
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

# This is a simplified demonstration of how to optimize a Reaktoro with Pyomo

def minimize_co2_dose():
    # create and solve initial state in Reaktoro
    db = rkt.PhreeqcDatabase("pitzer.dat")
    temp = 298.15 * pyo.units.K
    pres = 101325 * pyo.units.Pa
    initial_ph = None
    target_ph = 6 * pyo.units.dimensionless
    aq_species = {
        "H2O": 1 * pyo.units.kg,
        "Ca+2": 0 * pyo.units.mol,
    }
    gas_species = {
        "CO2(g)": 0 * pyo.units.mol,
        "H2O(g)": 0 * pyo.units.mol,
    }
    solid_species = None
    state = build_rkt_state(db, aq_species, gas_species, solid_species, temp, pres, allow_solids=False)
    j = update_rkt_state(state, temp, pres, initial_ph)
    #print(state)
    print(j)

    # build model
    model = build_pyo_model(state)
    # add variables to model
    model.temperature = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.K)
    model.pressure = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.Pa)
    model.ph = pyo.Var(bounds=[0,14], initialize=0, units=pyo.units.dimensionless)
    model.species_amounts = pyo.Var(model.all_species, domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.mol)
    update_pyo_model(state, model)

    external_model = ReaktoroOutputModel()
    external_model.get_model_inputs(model)
    external_model.get_rkt_state(state)
    model.reactor = ExternalGreyBoxBlock(external_model=external_model)

    # optimize model (minimize CO2 addition to reach target pH)
    model.ph_constraint = pyo.Constraint(expr=(model.reactor.inputs['ph'] == target_ph))
    CO2_init = model.species_amounts["CO2(g)"]
    model.CO2_dose = pyo.Objective(expr=(model.reactor.inputs[r"species_amounts['CO2(g)']"] - CO2_init), sense=pyo.minimize)

    solver = pyo.SolverFactory("cyipopt")
    results = solver.solve(model)
    pyo.assert_optimal_termination(results)

    model.pprint()
    return model

    '''
    # costing
    costs = {"slaked_lime": 160, "soda_ash": 134}
    mws = {"slaked_lime": 74.093, "soda_ash": 105.998}
    pyo.units.load_definitions_from_strings(['USD = [currency]'])
    amendments = pyo.Set(initialize=list(costs.keys()))
    model.amendment_costs = pyo.Var(
        amendments, domain=pyo.NonNegativeReals, initialize=costs, units=pyo.units.USD/pyo.units.t)
    model.amendment_mw = pyo.Var(
        amendments, domain=pyo.NonNegativeReals, initialize=mws, units=pyo.units.g/pyo.units.mol)

    # lime amendment stoichiometry constraint
    def _stoich_rule(stoich, d1, d2):
        return d1 - stoich * d2 == 0

    molal = pyo.units.mol/pyo.units.kg
    model.dose_OH = pyo.Var(domain=pyo.NonNegativeReals, initialize=0., units=molal)
    model.dose_Ca = pyo.Var(domain=pyo.NonNegativeReals, initialize=0., units=molal)
    model.lime_stoich = pyo.Constraint(rule=_stoich_rule(2, model.dose_Ca, model.dose_OH))

    # soda amendment stoichiometry constraint
    model.dose_Na = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=molal)
    model.dose_CO3 = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.mol/pyo.units.kg)
    model.soda_stoich = pyo.Constraint(expr = model.dose_Na - 2. * model.dose_CO3 == 0. * molal)

    # ph constraint
    ph_lb = 10
    ph_ub = 11
    model.softening_ph = pyo.Constraint(rule=(ph_lb, model.inlet.ph, ph_ub))

    # alkalinity constraint
    residual_alk_ub = 1e-5 * pyo.units.mol/pyo.units.kg
    model.residual_alk = pyo.Constraint(expr = model.inlet.alkalinity <= residual_alk_ub)

    # maybe replace survey with bisection solve
    survey = {}
    surveys = {
        "Ca+2": [.01, .03, .05],
        "CO3-2": [.02, .04, .08],
    }
    u = pyo.units.mol/pyo.units.kg
    for idx, step in enumerate(itertools.product(*surveys.values())):
        survey[idx] = {"Ca+2": step[0] * u, "CO3-2": step[1] * u}

    # optimization objective
    def _obj_fn(model):
        lime_cost = model.amendment_costs["slaked_lime"] / model.amendment_mw["slaked_lime"] * model.dose_Ca
        soda_cost = model.amendment_costs["soda_ash"] / model.amendment_mw["soda_ash"] * model.dose_CO3
        return lime_cost + soda_cost
    model.softening_cost = pyo.Objective(rule=_obj_fn, sense=pyo.minimize)

    # optimize model
    opt = pyo.SolverFactory("ipopt")

    models = [model]
    inlet_specs = helpers.setup_equilibrium_problem(inlet_state)
    outlet_specs = helpers.setup_equilibrium_problem(outlet_state)

    for step, amt in survey.items():
        print(f"Step {step} ...")
        model_clone = model.clone()
        inlet_state_clone = rkt.ChemicalState(inlet_state)
        outlet_state_clone = rkt.ChemicalState(outlet_state)

        model_clone.dose_Ca.fix(amt["Ca+2"])
        rkt_unit = str(pyo.units.get_units(model_clone.dose_Ca) * pyo.units.get_units(model_clone.inlet.water_mass))
        inlet_state_clone.add("Ca+2", pyo.value(model_clone.dose_Ca) * pyo.value(model_clone.inlet.water_mass), rkt_unit)
        print(f"- Ca+2 dose: {pyo.value(model_clone.dose_Ca)} {rkt_unit}")
        inlet_state_clone.add("OH-", 2 * pyo.value(model_clone.dose_Ca * pyo.value(model_clone.inlet.water_mass)), rkt_unit)
        print(f"- OH- dose: {2 * pyo.value(model_clone.dose_Ca)} {rkt_unit}")

        # soda dose
        model_clone.dose_CO3.fix(amt["CO3-2"])
        inlet_state_clone.add("Na+", 2 * pyo.value(model_clone.dose_CO3) * pyo.value(model_clone.inlet.water_mass), rkt_unit)
        print(f"- Na+ dose: {2 * pyo.value(model_clone.dose_CO3)} {rkt_unit}")
        inlet_state_clone.add("CO3-2", pyo.value(model_clone.dose_CO3) * pyo.value(model_clone.inlet.water_mass), rkt_unit)
        print(f"- CO3-2 dose: {pyo.value(model_clone.dose_CO3)} {rkt_unit}")

        # get variable data from Reaktoro and add to unit models
        helpers.solve_equilibrium_problem(inlet_state_clone, inlet_specs)
        helpers.solve_equilibrium_problem(outlet_state_clone, outlet_specs)
        helpers.get_vars(model_clone.inlet, inlet_state_clone)
        helpers.get_vars(model_clone.outlet, outlet_state_clone)

        print()
        print(f"Cost of softening = ${round(pyo.value(model_clone.softening_cost), 2)} per kg of water treated")
        models.append(model_clone)

        #model_clone.display()

    softening_costs = [pyo.value(model.softening_cost) for model in models[1:]]
    for idx, cost in enumerate(softening_costs):
        if cost == min(softening_costs):
            minimum_cost = cost # per kg

    print()
    print(f"Minimum softening cost: ${minimum_cost} per kg of water treated")
    '''

if __name__ == "__main__":
    model = minimize_co2_dose()
