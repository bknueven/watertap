import math
import numpy as np
import reaktoro as rkt
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from wt_rkt_helpers import ReaktoroOutputModel, build_rkt_state, solve_eq_problem, build_pyo_model

"""
This file demonstrates a simple example of integration between Pyomo and Reaktoro.

The three components of this file are:
    - A function to get a standalone equilibrium solution and create an initial Reaktoro state and Pyomo model
    - A plotting function of pH vs. CO2 dose for a sweep of CO2 doses (at fixed temperature and pressure)
    - A function to minimize CO2 dose given a target pH using the External Grey Box model from Pyomo
"""

def initial_solve(CO2_init, temp, pres):
    db = rkt.PhreeqcDatabase("pitzer.dat")
    aq_species = {
        "H2O": 1 * pyo.units.kg,
        "CO2": CO2_init * pyo.units.mol,
    }
    state = build_rkt_state(
        db, aq_species, gas_species=None, solid_species=None, temp=temp, pres=pres, allow_solids=False,
    )
    rkt_inputs = {
        "temperature": pyo.value(temp),
        "pressure": pyo.value(pres),
        "CO2": CO2_init,
    }
    solve_eq_problem(state, params=rkt_inputs)

    # create initial Pyomo model
    rkt_outputs = [species.name() for species in state.system().species()]
    model = build_pyo_model(state, rkt_inputs, rkt_outputs)
    return state, model

def plot_CO2_sweep(CO2_init, CO2_range, temp, pres):
    # create a dictionary with CO2 dose (mol) vs. pH (-)
    state, model = initial_solve(CO2_init, temp, pres)
    xy_dict = {}
    for amount in CO2_range:
        state.set("CO2", amount, "mol")
        solve_eq_problem(state)
        xy_dict[amount] = -math.log10(state.props().speciesActivity("H+").val())
    #print(xy_dict)
    plt.title("pH change with CO2 addition")
    plt.plot(xy_dict.keys(), xy_dict.values())
    plt.ylabel("pH (-)")
    plt.xlabel("CO2 amount (mol)")
    plt.grid("both")

def minimize_co2_dose(CO2_init, target_ph, temp, pres):
    # create initial state in Reaktoro
    state, model = initial_solve(CO2_init, temp, pres)

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

    # fix temperature and pressure, CO2 is unbound
    model.temp_constraint = pyo.Constraint(expr=model.reactor.inputs['temperature'] == pyo.value(temp))
    model.pres_constraint = pyo.Constraint(expr=model.reactor.inputs['pressure'] == pyo.value(pres))

    # add model objective
    model.CO2_dose = pyo.Objective(expr=model.reactor.inputs["CO2"], sense=pyo.minimize)

    # solve model with cyipopt
    solver = pyo.SolverFactory("cyipopt")
    solver.config.options["bound_relax_factor"] = 0.0
    solver.config.options['hessian_approximation'] = 'limited-memory'
    results = solver.solve(model, tee=False)
    pyo.assert_optimal_termination(results)

    for species in model.equilibrium_species:
        model.equilibrium_species[species].fix(state.props().speciesActivity(species).val())
    model.solution_pH.fix(-pyo.log10(model.equilibrium_species["H+"]))
    return state, model


if __name__ == "__main__":
    # set problem bounds
    CO2_init = 1e-8
    CO2_limit = 1
    # sweep over 1000 CO2 concentrations for plot
    CO2_range = np.linspace(CO2_init, CO2_limit, 1000)

    # function to quickly get pH from Reaktoro
    ph = lambda state: rkt.AqueousProps(state.props()).pH().val()

    # initial state vars
    temp = 298.15 * pyo.units.K
    pres = 101325 * pyo.units.Pa

    # plot pH vs CO2 sweep
    plot_CO2_sweep(CO2_init, CO2_range, temp, pres)

    # Solve standalone equilibrium problem given a guess for CO2 amount
    CO2_init = .25
    state, model = initial_solve(CO2_init, temp, pres)
    print(f"Standalone solve pH: {ph(state)}")
    print(f"CO2 amount: {CO2_init} mol\n")

    # Optimize CO2 amount given a target pH
    CO2_init = .1
    target_ph = 3.5
    state, model = minimize_co2_dose(CO2_init, target_ph, temp, pres)
    CO2_amount = pyo.value(model.reactor.inputs["CO2"])
    print(f"Grey box solve pH: {ph(state)}")
    print(f"CO2 amount: {CO2_amount} mol\n")
