import math
import numpy as np
import reaktoro as rkt
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from wt_rkt_helpers import ReaktoroOutputModel, build_rkt_state, solve_eq_problem, build_pyo_model

"""
This file demonstrates a complex example of integration between Pyomo and Reaktoro.

The four components of this file are:
    - A function for problem setup
    - A function to get a standalone equilibrium solution and create an initial Reaktoro state and Pyomo model
    - A plotting function of pH vs. CO2 dose for a sweep of CO2 doses (at fixed temperature and pressure)
    - A function to minimize Lime, Soda, and MgOx doses given a target alkalinity using the External Grey Box model from Pyomo
"""

def problem_setup(db, temp, pres):
    # Using BGW5
    initial_feed = {
        "H2O": 1e3 * pyo.units.g,
        "H+": 0 * pyo.units.g,
        "Na+": 1.120 * pyo.units.g,
        "CO3-2": .250 * pyo.units.g,
        "K+": .015 * pyo.units.g,
        "Ca+2": .150 * pyo.units.g,
        "OH-": 0 * pyo.units.g,
        "Mg+2": .033 * pyo.units.g,
        "Cl-": 1.750 * pyo.units.g,
        "SO4-2": .260 * pyo.units.g,
        "HCO3-": .250 * pyo.units.g,
        "SiO2": .0305 * pyo.units.g,
    }

    # NOTE: SiO2 is not represented in Pitzer database, so a conversion to mg of H4SiO4 is required
    def _convert_species(feed, species_map, mw_map, stoich_map, db):
        # Convert a species to another species and conserve mass of target element
        # i.e., SiO2 -> H4SiO4
        # species_map = {SiO2: H4SiO4} # species to convert
        # mw_map = {SiO2: 60.0843, H4SiO4: 96.1163} # g/mol of species
        # stoich_map = {SiO2: 1, H4SiO4: 1} # 1 mol of Si per mol of species
        for species, conv_species in species_map.items():
            init_species = feed.pop(species)
            init_species_mass = pyo.value(init_species)
            units = pyo.units.get_units(init_species)
            element_amount = init_species_mass / mw_map[species] * stoich_map[species]
            conv_species_amount = stoich_map[conv_species] * element_amount
            conv_species_mass = conv_species_amount * mw_map[conv_species]
            feed[conv_species] = conv_species_mass * units
        return

    _convert_species(
        initial_feed, {"SiO2": "H4SiO4"}, {"SiO2": 60.0843, "H4SiO4": 96.1163}, {"SiO2": 1, "H4SiO4": 1}, db
    )

    # specify solids likely to form in this scenario
    solid_species = {"Calcite": 0, "Aragonite": 0, "Gypsum": 0, "SiO2(a)": 0, "Magnesite": 0, "Dolomite": 0}
    # build state and solve initial equilibrium problem
    state = build_rkt_state(
        db, initial_feed, gas_species=None, solid_species=solid_species, temp=temp, pres=pres, allow_solids=True,
    )
    solve_eq_problem(state)
    return state

def plot_softening_sweep(state, lime_range, soda_range, mgox_range):
    # create a dictionary with softener amount (mol) vs. alkalinity (eq/L)
    plt.title("Alkalinity change with multi-species sweep")
    plt.ylabel("alkalinity (eq/L)")
    plt.xlabel("softening agent (mol)")
    plt.grid("both")
    initial_state = rkt.ChemicalState(state)
    aq_props = rkt.AqueousProps(state.props())
    softening_species = ["Ca+2", "Mg+2", "OH-", "Na+", "CO3-2"]
    species_amount = {species: state.speciesAmount(species).val() for species in softening_species}
    # plot softening agents independently
    xy_dict = {}
    working_state = rkt.ChemicalState(initial_state)
    for amount in lime_range:
        working_state.set("Ca+2", species_amount["Ca+2"] + amount, "mol")
        working_state.set("OH-", species_amount["OH-"] + 2*amount, "mol")
        solve_eq_problem(working_state)
        aq_props.update(working_state)
        alkalinity = aq_props.alkalinity().val()
        xy_dict[amount] = alkalinity
    plt.plot(xy_dict.keys(), xy_dict.values(), label="lime")

    xy_dict = {}
    working_state = rkt.ChemicalState(initial_state)
    for amount in soda_range:
        working_state.set("Na+", species_amount["Na+"] + 2*amount, "mol")
        working_state.set("CO3-2", species_amount["CO3-2"] + amount, "mol")
        solve_eq_problem(working_state)
        aq_props.update(working_state)
        alkalinity = aq_props.alkalinity().val()
        xy_dict[amount] = alkalinity
    plt.plot(xy_dict.keys(), xy_dict.values(), label="soda")

    xy_dict = {}
    working_state = rkt.ChemicalState(initial_state)
    for amount in mgox_range:
        working_state.set("Mg+2", species_amount["Mg+2"] + amount, "mol")
        working_state.set("OH-", species_amount["OH-"] + 2*amount, "mol")
        solve_eq_problem(working_state)
        aq_props.update(working_state)
        alkalinity = aq_props.alkalinity().val()
        xy_dict[amount] = alkalinity
    plt.plot(xy_dict.keys(), xy_dict.values(), label="mgox")
    plt.legend()

def standalone_softening_solver(state, temp, pres, lime_dose, soda_dose, mgox_dose):
    #solve an equilibrium problem given softener doses (mol)
    softening_species = ["Ca+2", "Mg+2", "OH-", "Na+", "CO3-2"]
    species_amount = {species: state.speciesAmount(species).val() for species in softening_species}
    state.set("Ca+2", species_amount["Ca+2"] + pyo.value(lime_dose), "mol")
    state.set("Mg+2", species_amount["Mg+2"] + pyo.value(mgox_dose), "mol")
    state.set("OH-", species_amount["OH-"] + 2 * (pyo.value(lime_dose) + pyo.value(mgox_dose)), "mol")
    state.set("Na+", species_amount["Na+"] + 2 * pyo.value(soda_dose), "mol")
    state.set("CO3-2", species_amount["CO3-2"] + pyo.value(soda_dose), "mol")
    solve_eq_problem(state)

def minimize_softener_dose(state, lime_init, soda_init, mgox_init, target_ph, temp, pres):
    # specify Reaktoro inputs/outputs for use with Pyomo Grey Box
    external_model_inputs = {
        "temperature": pyo.value(temp),
        "pressure": pyo.value(pres),
        #"Ca+2": pyo.value(lime_init),
        #"Mg+2": pyo.value(mgox_init),
        #"OH-": 2 * (pyo.value(lime_init) + pyo.value(mgox_init)),
        "Na+": 2 * pyo.value(soda_init),
        "CO3-2": pyo.value(soda_init),
    }
    external_model_outputs = [species.name() for species in state.system().species()]

    # create initial Pyomo model
    model = build_pyo_model(state, external_model_inputs, external_model_outputs)

    # create variables for dosants
    #model.lime_dose = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.mol)
    model.soda_dose = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.mol)
    #model.mgox_dose = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.mol)

    # constrain pH such that solution pH == target pH
    model.target_pH = pyo.Var(bounds=[0,14], initialize=7, units=pyo.units.dimensionless)
    model.solution_pH = pyo.Var(bounds=[0,14], initialize=7, units=pyo.units.dimensionless)
    model.target_pH.fix(target_ph)
    # pH is an estimation; model.reactor.outputs are roughly equivalent to Molarity (mol/L)
    # need to find a way to implement the below transformation:
    # molarity = model.reactor.outputs["H+"] / state.props().volume().val()
    # activity = state.props().speciesActivities("H+").val() * molarity
    # model.solution_pH = -pyo.log10(activity)
    model.pH_constraint = pyo.Constraint(expr=model.target_pH == -pyo.log10(model.reactor.outputs['H+']))

    # fix temperature and pressure, fix stoichiometric ratios for softening agents
    model.temp_constraint = pyo.Constraint(expr=model.reactor.inputs['temperature'] == pyo.value(temp))
    model.pres_constraint = pyo.Constraint(expr=model.reactor.inputs['pressure'] == pyo.value(pres))
    model.soda_constraint = pyo.Constraint(expr=model.reactor.inputs['Na+'] == 2*model.reactor.inputs['CO3-2'])

    # add model objective
    def alkalinity_obj(model, state):
        mol_alk = 0
        mol_alk += model.reactor.outputs["Na+"]
        mol_alk += model.reactor.outputs["K+"]
        mol_alk += 2*model.reactor.outputs["Ca+2"]
        mol_alk += 2*model.reactor.outputs["Mg+2"]
        return mol_alk
    model.alk_obj = pyo.Objective(expr=model.reactor.outputs['SiO2(a)'], sense=pyo.maximize)

    # solve model with cyipopt
    solver = pyo.SolverFactory("cyipopt")
    solver.config.options["bound_relax_factor"] = 0.0
    solver.config.options['hessian_approximation'] = 'limited-memory'
    #results = solver.solve(model, tee=False)
    #pyo.assert_optimal_termination(results)

    # update model with equilibrium activities and solution pH
    for species in model.output_activity:
        model.output_activity[species].fix(state.props().speciesActivity(species).val())
    model.solution_pH.fix(-pyo.log10(model.output_activity["H+"]))
    return model

if __name__ == "__main__":
    # use Pitzer database from PHREEQC
    db = rkt.PhreeqcDatabase("pitzer.dat")

    # initial state vars
    temp = 298.15 * pyo.units.K
    pres = 101325 * pyo.units.Pa

    # see problem formulation within this function
    initial_state = problem_setup(db, temp, pres)

    # plot sweep of softening agent amounts to get a rough idea of optimal solution
    lime_range = np.linspace(0, 1e-4, 100)
    soda_range = np.linspace(0, 2e-4, 100)
    mgox_range = np.linspace(0, 1e-4, 100)
    working_state = rkt.ChemicalState(initial_state)
    plot_softening_sweep(working_state, lime_range, soda_range, mgox_range)

    # function to quickly get values from Reaktoro
    si = lambda state: state.props().elementAmount('Si').val()
    sio2 = lambda state: state.props().speciesAmount('SiO2(a)').val()
    alk = lambda state: rkt.AqueousProps(state.props()).alkalinity().val()
    ph = lambda state: rkt.AqueousProps(state.props()).pH().val()


    # solve standalone equilibrium problem with guesses for softening agent amounts
    # note: softening agents are expressed as totals due potential presence of ions in the initial feed
    lime_total = 1e-4 * pyo.units.mol
    soda_total = 2e-4 * pyo.units.mol
    mgox_total = 1e-4 * pyo.units.mol
    working_state = rkt.ChemicalState(initial_state)
    standalone_softening_solver(working_state, temp, pres, lime_total, soda_total, mgox_total)
    print(f"Standalone solution Alk removal: {round(abs(alk(working_state) - alk(initial_state)), 5)} eq/L")
    print(f"Standalone solution Si removal: {round(sio2(working_state) / si(initial_state), 1) * 100} %")
    print(f"Standalone solution pH: {round(ph(working_state), 2)}")
    print(f"lime amount: {pyo.value(lime_total)} mol")
    print(f"soda amount: {pyo.value(soda_total)} mol")
    print(f"mgox amount: {pyo.value(mgox_total)} mol\n")

    # Optimize alkalinity given a target pH
    lime_init = 1e-5 * pyo.units.mol
    soda_init = 1e-5 * pyo.units.mol
    mgox_init = 1e-5 * pyo.units.mol
    target_ph = 10
    working_state = rkt.ChemicalState(initial_state)
    model = minimize_softener_dose(working_state, lime_init, soda_init, mgox_init, target_ph, temp, pres)
    print(f"Grey Box solution Si removal : {sio2(working_state) / si(initial_state) * 100} %")
    print(f"Grey Box solution pH: {ph(working_state)}")
    #print(f"lime amount: {pyo.value(model.reactor.inputs['Ca+2'])} mol")
    print(f"soda amount: {pyo.value(model.reactor.inputs['CO3-2'])} mol")
    #print(f"mgox amount: {pyo.value(model.reactor.inputs['Mg+2'])} mol\n")
