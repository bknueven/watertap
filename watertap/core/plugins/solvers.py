#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
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
import math
import time

import numpy as np

import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.solvers.plugins.solvers.IPOPT import IPOPT

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


@pyo.SolverFactory.register(
    "ipopt-watertap",
    doc="The Ipopt NLP solver, with user-based variable and automatic Jacobian constraint scaling",
)
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
        if "acceptable_constr_viol_tol" not in self.options:
            self.options["acceptable_constr_viol_tol"] = 1e-08
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
            # run Constraint Consensus Algorithm to refine
            # the initial point given by the user
            run_cca(self._model, nlp, self._tee)
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


## reconfigure IDAES to use the ipopt-watertap solver
import idaes

_default_solver_config_value = idaes.cfg.get("default_solver")
_idaes_default_solver = _default_solver_config_value._default

_default_solver_config_value.set_default_value("ipopt-watertap")
if not _default_solver_config_value._userSet:
    _default_solver_config_value.reset()


from pyomo.environ import ComponentMap
from pyomo.contrib.incidence_analysis import get_incident_variables, IncidenceMethod


def run_cca(model, nlp, tee, tol=1e-08):
    """
    Constraint Consensus Algorithm: Attempts to refine an initial point.

    This implementation mostly reflects the following reference:
        Chinneck, John W. "The constraint consensus method for finding approximately
        feasible points in nonlinear programs." INFORMS journal on computing 16.3
        (2004): 255-265.
    """

    start_time = time.time()

    alpha_tol = 1e01
    beta_tol = 1e00
    iter_limit = 100

    if tee:
        print("Initialization Refinement: Constraint Consensus Algorithm")
        print(
            "{_iter:<6}"
            "{infeasibility:<16}"
            "{alpha:<11}"
            "{beta:<11}"
            "{time:<6}".format(
                _iter="iter",
                infeasibility="   inf_pr   ",
                alpha="inf_dist",
                beta=" ||d|| ",
                time="time",
            )
        )

    lb = nlp.primals_lb()
    ub = nlp.primals_ub()

    x = nlp.init_primals().copy()

    ineq_lb = nlp.ineq_lb()
    ineq_ub = nlp.ineq_ub()

    primal_inf = np.inf
    alpha = np.nan
    beta = 0.0

    for _iter in range(iter_limit):
        # step 1
        ninf = 0
        n = np.zeros(len(x), dtype=int)
        s = np.zeros(len(x), dtype=float)

        # step 2
        nlp.set_primals(x)

        eq_val = nlp.evaluate_eq_constraints()
        ineq_val = nlp.evaluate_ineq_constraints()

        more_votes = len(eq_val) + len(ineq_val)

        prior_primal_inf = primal_inf
        if eq_val.size == 0:
            max_eq_resid = 0
        else:
            max_eq_resid = np.max(np.abs(eq_val))
        if ineq_val.size == 0:
            max_ineq_resid = 0
        else:
            max_lb_resid = np.max(ineq_lb - ineq_val)
            max_ub_resid = np.max(ineq_val - ineq_ub)
            max_ineq_resid = max(max_lb_resid, max_ub_resid)
        max_lb_resid = np.max(lb - x)
        max_ub_resid = np.max(x - ub)
        max_bound_resid = max(max_lb_resid, max_ub_resid)
        primal_inf = max(max_eq_resid, max_ineq_resid, max_bound_resid)

        # safeguard -- if we go off the rails
        if prior_primal_inf * 1e6 < primal_inf:
            nlp.set_primals(x - t)
            break

        alpha = 0.0
        jac_eq = nlp.evaluate_jacobian_eq().tocsr()
        jac_ineq = nlp.evaluate_jacobian_ineq().tocsr()

        alpha_tol_satisfied = True
        for idx, viol in enumerate(eq_val):
            if viol == 0.0:
                # constraint is exactly satisfied
                continue

            #  viol * jac_eq[idx] / ||jac_eq[idx]||**2
            row = jac_eq.getrow(idx)
            div = sum(val * val for val in row.data)
            feas_dis = abs(viol / math.sqrt(div))
            alpha = max(alpha, feas_dis)
            # always update constraints
            if feas_dis > alpha_tol:
                alpha_tol_satisfied = False
            if abs(viol) < tol:
                n[row.indices] += more_votes
                row *= -(viol / div) * more_votes
            else:
                n[row.indices] += 1
                row *= -(viol / div)
            ninf += 1
            s += row

        for idx, val in enumerate(ineq_val):
            viol_lb = ineq_lb[idx] - val
            viol_ub = val - ineq_ub[idx]
            if max(viol_lb, viol_ub) < 0.0:
                # try to keep an interior point by making
                # these inequalities strict
                continue
            if viol_lb > viol_ub:
                viol = viol_lb
                d = 1
            else:
                viol = viol_ub
                d = -1
            #  viol * jac_eq[idx] / ||jac_eq[idx]||**2
            row = jac_ineq.getrow(idx)
            div = sum(val * val for val in row.data)
            feas_dis = abs(viol / math.sqrt(div))
            alpha = max(alpha, feas_dis)
            # always update inequality constraints
            if feas_dis > alpha_tol:
                alpha_tol_satisfied = False
            if abs(viol) < tol:
                n[row.indices] += more_votes
                row *= d * (viol / div) * more_votes
            else:
                row *= d * (viol / div)
                n[row.indices] += 1
            ninf += 1
            s += row

        # turn single row-matrix
        # back into array
        if ninf > 0:
            s = s.A[0]
        for idx, val in enumerate(x):
            viol_lb = lb[idx] - val
            viol_ub = val - ub[idx]
            if max(viol_lb, viol_ub) < 0.0:
                # try to keep an interior point by making
                # these inequalities strict
                continue
            # push variable bounds inside
            if viol_lb > viol_ub:
                viol = 1.1 * viol_lb
                d = 1
            else:
                viol = 1.1 * viol_ub
                d = -1
            alpha = max(alpha, viol)
            # always update variable bounds
            if viol > alpha_tol:
                alpha_tol_satisfied = False
            ninf += 1
            # bounds get the votes!
            if abs(viol) > 0:
                s[idx] += d * viol * more_votes * more_votes
                n[idx] += more_votes * more_votes
                print(f"violated bound of {viol} for var {nlp.vlist[idx].name}")
            else:
                s[idx] += d * viol * more_votes
                n[idx] += more_votes

        if tee:
            print(
                "{_iter:<6}"
                "{infeasibility:<16.7e}"
                "{alpha:<11.2e}"
                "{beta:<11.2e}"
                "{time:<7.2f}".format(
                    _iter=_iter,
                    infeasibility=primal_inf,
                    alpha=alpha,
                    beta=beta,
                    time=time.time() - start_time,
                )
            )

        # step 3
        if alpha_tol_satisfied:
            break

        # step 4
        t = np.zeros(len(x), dtype=float)
        for i, (s_i, n_i) in enumerate(zip(s, n)):
            if n_i != 0:
                t[i] = s_i / n_i

        # step 5
        beta = np.linalg.norm(t)
        if beta <= beta_tol:
            break

        # step 6 & 7
        x = x + t

        # step 8
        continue

    primals = nlp.get_primals()
    for idx, v in enumerate(nlp.vlist):
        v.set_value(primals[idx], skip_validation=True)
