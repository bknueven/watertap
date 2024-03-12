#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import logging

import pyomo.environ as pyo
from pyomo.common.collections import Bunch
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.solvers.plugins.solvers.IPOPT import IPOPT
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.solver.util import get_objective

from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus

import idaes.core.util.scaling as iscale
from idaes.core.util.scaling import (
    get_scaling_factor,
    set_scaling_factor,
    unset_scaling_factor,
)
from idaes.logger import getLogger

_log = getLogger("watertap.core")

_pyomo_nl_writer_log = logging.getLogger("pyomo.repn.plugins.nl_writer")


def _pyomo_nl_writer_logger_filter(record):
    msg = record.getMessage()
    if "scaling_factor" in msg and "model contains export suffix" in msg:
        return False
    return True


class IpoptWaterTAP(IPOPT):
    def __init__(self, **kwds):
        kwds["name"] = "ipopt-watertap"
        self._cleanup_needed = False
        super().__init__(**kwds)

    def _presolve(self, *args, **kwds):
        if len(args) > 1 or len(args) == 0:
            raise TypeError(
                f"IpoptWaterTAP.solve takes 1 positional argument but {len(args)} were given"
            )
        if not isinstance(args[0], (_BlockData, IBlock)):
            raise TypeError(
                "IpoptWaterTAP.solve takes 1 positional argument: a Pyomo ConcreteModel or Block"
            )

        # until proven otherwise
        self._cleanup_needed = False

        self._tee = kwds.get("tee", False)

        # Set the default watertap options
        if "tol" not in self.options:
            self.options["tol"] = 1e-08
        if "constr_viol_tol" not in self.options:
            self.options["constr_viol_tol"] = 1e-08
        if "bound_relax_factor" not in self.options:
            self.options["bound_relax_factor"] = 0.0
        if "honor_original_bounds" not in self.options:
            self.options["honor_original_bounds"] = "no"

        if not self._is_user_scaling():
            super()._presolve(*args, **kwds)
            self._cleanup()
            return

        if self._tee:
            print(
                "ipopt-watertap: Ipopt with user variable scaling and IDAES jacobian constraint scaling"
            )

        # These options are typically available with gradient-scaling, and they
        # have corresponding options in the IDAES constraint_autoscale_large_jac
        # function. Here we use their Ipopt names and default values, see
        # https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_NLP_Scaling
        max_grad = self._get_option("nlp_scaling_max_gradient", 100)
        min_scale = self._get_option("nlp_scaling_min_value", 1e-08)

        # These options are custom for the IDAES constraint_autoscale_large_jac
        # function. We expose them as solver options as this has become part
        # of the solve process.
        ignore_variable_scaling = self._get_option("ignore_variable_scaling", False)
        ignore_constraint_scaling = self._get_option("ignore_constraint_scaling", False)

        self._model = args[0]
        self._cache_scaling_factors()
        self._cleanup_needed = True
        _pyomo_nl_writer_log.addFilter(_pyomo_nl_writer_logger_filter)

        # NOTE: This function sets the scaling factors on the
        #       constraints. Hence we cache the constraint scaling
        #       factors and reset them to their original values
        #       so that repeated calls to solve change the scaling
        #       each time based on the initial values, just like in Ipopt.
        try:
            _, _, nlp = iscale.constraint_autoscale_large_jac(
                self._model,
                ignore_constraint_scaling=ignore_constraint_scaling,
                ignore_variable_scaling=ignore_variable_scaling,
                max_grad=max_grad,
                min_scale=min_scale,
            )
        except Exception as err:
            nlp = None
            if str(err) == "Error in AMPL evaluation":
                print(
                    "ipopt-watertap: Issue in AMPL function evaluation; Jacobian constraint scaling not applied."
                )
                halt_on_ampl_error = self.options.get("halt_on_ampl_error", "yes")
                if halt_on_ampl_error == "no":
                    print(
                        "ipopt-watertap: halt_on_ampl_error=no, so continuing with optimization."
                    )
                else:
                    self._cleanup()
                    raise RuntimeError(
                        "Error in AMPL evaluation.\n"
                        "Run ipopt with halt_on_ampl_error=yes and symbolic_solver_labels=True to see the affected function."
                    )
            else:
                print("Error in constraint_autoscale_large_jac")
                self._cleanup()
                raise

        # set different default for `alpha_for_y` if this is an LP
        # see: https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_alpha_for_y
        if nlp is not None:
            if nlp.nnz_hessian_lag() == 0:
                if "alpha_for_y" not in self.options:
                    self.options["alpha_for_y"] = "bound-mult"

        try:
            # this creates the NL file, among other things
            return super()._presolve(*args, **kwds)
        except:
            self._cleanup()
            raise

    def _cleanup(self):
        if self._cleanup_needed:
            self._reset_scaling_factors()
            # remove our reference to the model
            del self._model
            _pyomo_nl_writer_log.removeFilter(_pyomo_nl_writer_logger_filter)

    def _postsolve(self):
        self._cleanup()
        return super()._postsolve()

    def _cache_scaling_factors(self):
        self._scaling_cache = [
            (c, get_scaling_factor(c))
            for c in self._model.component_data_objects(
                pyo.Constraint, active=True, descend_into=True
            )
        ]

    def _reset_scaling_factors(self):
        for c, s in self._scaling_cache:
            if s is None:
                unset_scaling_factor(c)
            else:
                set_scaling_factor(c, s)
        del self._scaling_cache

    def _get_option(self, option_name, default_value):
        # NOTE: options get reset to their original value at the end of the
        #       OptSolver.solve. The options in _presolve (where this is called)
        #       are already copies of the original, so it is safe to pop them so
        #       they don't get sent to Ipopt.
        option_value = self.options.pop(option_name, None)
        if option_value is None:
            option_value = default_value
        else:
            if self._tee:
                print(f"ipopt-watertap: {option_name}={option_value}")
        return option_value

    def _is_user_scaling(self):
        if "nlp_scaling_method" not in self.options:
            self.options["nlp_scaling_method"] = "user-scaling"
        if self.options["nlp_scaling_method"] != "user-scaling":
            if self._tee:
                print(
                    "The ipopt-watertap solver is designed to be run with user-scaling. "
                    f"Ipopt with nlp_scaling_method={self.options['nlp_scaling_method']} will be used instead"
                )
            return False
        return True


@pyo.SolverFactory.register(
    "ipopt-watertap",
    doc="The Ipopt NLP solver, with user-based variable and automatic Jacobian constraint scaling",
)
class IpoptWaterTAPFBBT:

    _base_solver = IpoptWaterTAP
    name = "ipopt-watertap"

    def __init__(self, **kwds):

        kwds["name"] = "ipopt-watertap-fbbt"
        self.options = Bunch()
        if kwds.get("options") is not None:
            for key in kwds["options"]:
                setattr(self.options, key, kwds["options"][key])

        self._bound_cache = pyo.ComponentMap()

    def executable(self):
        return self._base_solver().executable()

    def _cache_bounds(self, blk):
        self._bound_cache = pyo.ComponentMap()
        for v in blk.component_data_objects(pyo.Var, active=True, descend_into=True):
            # we could hit a variable more
            # than once because of References
            if v in self._bound_cache:
                continue
            self._bound_cache[v] = v.bounds

    def _restore_bounds(self):
        for v, bounds in self._bound_cache.items():
            v.bounds = bounds
        del self._bound_cache

    def _cache_active_constraints(self, blk):
        self._active_constraint_cache = []
        for c in blk.component_data_objects(
            pyo.Constraint, active=True, descend_into=True
        ):
            self._active_constraint_cache.append(c)

    def _restore_active_constraints(self):
        for c in self._active_constraint_cache:
            c.activate()

    def _fbbt(self, blk):
        try:
            fbbt(
                blk,
                feasibility_tol=1e-8,
                deactivate_satisfied_constraints=False,
            )
        except:
            # cleanup before raising
            self._restore_active_constraints()
            self._restore_bounds()
            raise
        all_fixed = True
        bound_relax_factor = 1e-6
        for v, (lb, ub) in self._bound_cache.items():
            if v.lb is not None and v.lb == v.ub:
                v.value = v.lb
            else:
                all_fixed = False
            if v.value is None:
                if v.lb is not None and v.ub is not None:
                    v.value = (v.lb + v.ub) / 2.0
            if lb is None:
                if v.lb is not None:
                    if v.value is None or v.value < v.lb:
                        v.value = v.lb
                    v.lb = v.lb - bound_relax_factor
            else:
                v.lb = max(lb, v.lb - bound_relax_factor)
            if ub is None:
                if v.ub is not None:
                    if v.value is None or v.value > v.ub:
                        v.value = v.ub
                    v.ub = v.ub + bound_relax_factor
            else:
                v.ub = min(ub, v.ub + bound_relax_factor)

        return all_fixed

    def solve(self, blk, *args, **kwds):

        solver = self._base_solver()

        for k, v in self.options.items():
            solver.options[k] = v

        self._cache_bounds(blk)
        self._cache_active_constraints(blk)

        try:
            all_fixed = self._fbbt(blk)
        except InfeasibleConstraintException:
            results = SolverResults()
            results.solver.status = SolverStatus.error
            results.solver.termination_condition = TerminationCondition.infeasible
            results.solver.termination_message = (
                "FBBT determined the model was infeasible"
            )
            results.solver.message = "FBBT proved infeasibility subject to tolerances."

            return results

        if all_fixed:
            obj = get_objective(blk)

            results = SolverResults()

            results.solver.status = SolverStatus.ok
            results.solver.termination_condition = TerminationCondition.optimal
            results.solver.termination_message = "FBBT solved the model"
            results.solver.message = "Solution found with interval arithmetic"

            if obj is None:
                results.problem.lower_bound = 0.0
                results.problem.upper_bound = 0.0
            else:
                results.problem.lower_bound = pyo.value(obj)
                results.problem.upper_bound = pyo.value(obj)

            solution = Solution()
            solution.status = SolutionStatus.optimal
            solution.gap = 0.0

            for v in self._bound_cache:
                solution.variable[v.name] = {"Value": v.value}
            if hasattr(blk, "dual") and blk.dual.import_enabled():
                _log.warning("Cannot get duals for model solved by FBBT")
            if hasattr(blk, "rc") and blk.rc.import_enabled():
                _log.warning("Cannot get reduced costs for model solved by FBBT")
            if hasattr(blk, "slack") and blk.slack.import_enabled():
                for c in self._active_constraint_cache:
                    solution.constraint[c.name] = {"Slack": 0}

            results.solution.insert(solution)

        else:  # FBBT could not solve it
            results = solver.solve(blk, *args, **kwds)

        self._restore_active_constraints()
        self._restore_bounds()

        return results


## reconfigure IDAES to use the ipopt-watertap solver
import idaes

_default_solver_config_value = idaes.cfg.get("default_solver")
_idaes_default_solver = _default_solver_config_value._default

_default_solver_config_value.set_default_value("ipopt-watertap")
if not _default_solver_config_value._userSet:
    _default_solver_config_value.reset()
