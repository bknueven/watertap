import math
import numpy as np
import reaktoro as rkt
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from wt_rkt_helpers import ReaktoroOutputModel, build_rkt_state, solve_eq_problem, build_pyo_model

"""
This file demonstrates a simple example of integration between Pyomo and Reaktoro.

The 3 components of this file are:
    - A plotting function of pH vs. CO2 amount for a sweep of CO2 amounts (at fixed temperature and pressure)
    - A function to get a standalone equilibrium solution given a total amount of CO2
    - A function to minimize CO2 dose given a target pH using the External Grey Box model from Pyomo
"""

# quickly get pH from Reaktoro
ph = lambda state: rkt.AqueousProps(state.props()).pH().val()

def plot_pH_dose_response(state, dosant, dose_range):
    # plot pH response w.r.t. dosant
    plt.title(f"pH response to {dosant}")
    plt.ylabel("pH (-)")
    plt.xlabel(f"{dosant} dose (mol)")
    plt.grid("both")
    xy_dict = {}
    initial_amount = state.props().speciesAmount(dosant).val()
    for dose in dose_range:
        state.set(dosant, initial_amount + dose, "mol")
        solve_eq_problem(state)
        xy_dict[dose] = ph(state)
    plt.plot(xy_dict.keys(), xy_dict.values(), label=dosant)
    plt.legend()

def standalone_solver(state, dosant, dose):
    # solve an equilibrium problem given a dosant (mol)
    initial_amount = state.props().speciesAmount(dosant).val()
    state.set(dosant, initial_amount + dose, "mol")
    solve_eq_problem(state)

def pyomo_optimizer(state, model, dosant, target_ph):
    # constrain pH such that solution pH == target pH
    model.target_pH = pyo.Var(bounds=[0,14], initialize=7, units=pyo.units.dimensionless)
    model.target_pH.fix(target_ph)
    # note: (possibly resolved) can get partial derivative from Reaktoro.EquilibriumSensitivity.dudw()
    # https://github.com/reaktoro/reaktoro/discussions/380
    # pH is an estimation; model.reactor.outputs are roughly equivalent to Molarity (mol/L)
    # need to find a way to implement the below transformation:
    # molarity = model.reactor.outputs["H+"] / state.props().volume().val()
    # activity = state.props().speciesActivities("H+").val() * molarity
    model.pH_constraint = pyo.Constraint(expr=model.target_pH == -pyo.log10(model.reactor.outputs['H+']))

    # fix temperature and pressure, CO2 is unbound
    model.temp_constraint = pyo.Constraint(
        expr=model.reactor.inputs['temperature'] == state.temperature().val(),
    )
    model.pres_constraint = pyo.Constraint(
        expr=model.reactor.inputs['pressure'] == state.pressure().val(),
    )

    # add model objective
    initial_amount = model.reactor.inputs[dosant]
    model.dose_objective = pyo.Objective(expr=model.reactor.inputs[dosant], sense=pyo.minimize)

    # solve model with cyipopt
    solver = pyo.SolverFactory("cyipopt")
    solver.config.options["bound_relax_factor"] = 0.0
    solver.config.options['hessian_approximation'] = 'limited-memory'
    results = solver.solve(model, tee=False)
    pyo.assert_optimal_termination(results)
    model.dose = pyo.Var(
        domain=pyo.NonNegativeReals, initialize=model.reactor.inputs[dosant]-initial_amount, units=pyo.units.mol,
    )
    model.pprint()
    return model

if __name__ == "__main__":
    # use Pitzer database from PHREEQC
    db = rkt.PhreeqcDatabase("pitzer.dat")

    # initial state vars
    temp = 298.15 * pyo.units.K
    pres = 101325 * pyo.units.Pa

    # set up Reaktoro state
    aq_species = {
        "H2O": 1 * pyo.units.kg,
        "CO2": 1e-16 * pyo.units.mol,
        "Ca+2": 1e-5 * pyo.units.mol,
    }
    # build state and solve initial equilibrium problem
    state = build_rkt_state(
        db, aq_species, gas_species=None, solid_species=None, temp=temp, pres=pres, allow_solids=False,
    )
    solve_eq_problem(state)

    working_state = rkt.ChemicalState(state)
    # Method 1: sweep through CO2 doses to get a rough idea of optimal solution
    dose_range = np.linspace(1e-8, 1., 1000)
    # plot pH vs CO2 sweep
    plot_pH_dose_response(working_state, "CO2", dose_range)

    # Method 2: solve standalone equilibrium problem with guess for CO2 dose
    working_state = rkt.ChemicalState(state)
    dose = 0.22
    standalone_solver(working_state, "CO2", dose)
    print(f"Standalone solution pH: {ph(working_state)}")
    print(f"CO2 dose: {dose} mol\n")

    # Method 3: optimize CO2 dose given a target pH
    working_state = rkt.ChemicalState(state)
    # specify Reaktoro inputs/outputs for use with Pyomo Grey Box
    # look in wt_rkt.helpers.py to see list of valid outputs (in _get_jacobian_rows)
    rkt_input = {"temperature": pyo.value(temp), "pressure": pyo.value(pres), "CO2": 0}
    rkt_output = "speciesActivity"
    # setup Pyomo model
    model = build_pyo_model(state, rkt_input, rkt_output)
    pyomo_optimizer(working_state, model, "CO2", target_ph=3.5)
    print(f"Grey Box solution pH: {ph(working_state)}")
    print(f"CO2 dose: {pyo.value(model.dose)} mol\n")

    # NOTE: on Method 3 I started getting this message:
    '''
    Exception ignored in: <function AmplInterface.__del__ at 0x0000027B9D3EF5E0>
    Traceback (most recent call last):
      File "~\AppData\Local\miniconda3\envs\watertap-rkt\lib\site-packages\pyomo\contrib\pynumero\asl.py", line 312, in __del__
        self.ASLib.EXTERNAL_AmplInterface_free_memory(self._obj)
    AttributeError: 'NoneType' object has no attribute 'EXTERNAL_AmplInterface_free_memory'

    Traceback (most recent call last):

      File ~\AppData\Local\miniconda3\envs\watertap-rkt\lib\site-packages\spyder_kernels\py3compat.py:356 in compat_exec
        exec(code, globals, locals)

      File ~\documents\software\watertap\watertap\tools\oli_api\pyomo_carbonation.py:120
        pyomo_optimizer(working_state, model, "CO2", target_ph=3.5)

      File ~\documents\software\watertap\watertap\tools\oli_api\pyomo_carbonation.py:71 in pyomo_optimizer
        results = solver.solve(model, tee=False)

      File ~\AppData\Local\miniconda3\envs\watertap-rkt\lib\site-packages\pyomo\contrib\pynumero\algorithms\solvers\cyipopt_solver.py:337 in solve
        nlp = pyomo_grey_box.PyomoNLPWithGreyBoxBlocks(model)

      File ~\AppData\Local\miniconda3\envs\watertap-rkt\lib\site-packages\pyomo\contrib\pynumero\interfaces\pyomo_grey_box_nlp.py:63 in __init__
        self._pyomo_nlp = PyomoNLP(pyomo_model)

      File ~\AppData\Local\miniconda3\envs\watertap-rkt\lib\site-packages\pyomo\contrib\pynumero\interfaces\pyomo_nlp.py:105 in __init__
        super(PyomoNLP, self).__init__(nl_file)

      File ~\AppData\Local\miniconda3\envs\watertap-rkt\lib\site-packages\pyomo\contrib\pynumero\interfaces\ampl_nlp.py:54 in __init__
        self._asl = _asl.AmplInterface(self._nl_file)

      File ~\AppData\Local\miniconda3\envs\watertap-rkt\lib\site-packages\pyomo\contrib\pynumero\asl.py:258 in __init__
        raise RuntimeError("Cannot load the PyNumero ASL interface (pynumero_ASL)")

    RuntimeError: Cannot load the PyNumero ASL interface (pynumero_ASL)
    '''
