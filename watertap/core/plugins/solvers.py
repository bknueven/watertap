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
from pyomo.core.base.PyomoModel import ModelSolutions
from pyomo.common.collections import Bunch
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.solver.util import get_objective
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

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
class IpoptWaterTAP:

    name = "ipopt-watertap"
    _base_solver = "ipopt_v2"

    def __init__(self, **kwds):
        kwds["name"] = self.name
        self.options = Bunch()
        if kwds.get("options") is not None:
            for key in kwds["options"]:
                setattr(self.options, key, kwds["options"][key])

    def executable(self):
        return pyo.SolverFactory(self._base_solver).config.executable.executable

    def solve(self, blk, *args, **kwds):

        solver = pyo.SolverFactory(self._base_solver)
        self._tee = kwds.get("tee", False)

        if get_objective(blk) is None:
            self._dummy_objective_name = unique_component_name(blk, "objective")
            blk.add_component(self._dummy_objective_name, pyo.Objective(expr=0))
        else:
            self._dummy_objective_name = None

        blk.solutions = ModelSolutions(blk)

        self._original_options = self.options

        self.options = dict()
        self.options.update(self._original_options)
        self.options.update(kwds.pop("options", {}))

        # Set the default watertap options
        if "tol" not in self.options:
            self.options["tol"] = 1e-08
        if "constr_viol_tol" not in self.options:
            self.options["constr_viol_tol"] = 1e-08
        if "bound_relax_factor" not in self.options:
            self.options["bound_relax_factor"] = 0.0
        if "honor_original_bounds" not in self.options:
            self.options["honor_original_bounds"] = "no"

        if self._tee:
            print(
                "ipopt-watertap: Ipopt with user variable scaling and IDAES jacobian constraint scaling"
            )

        nlp = PyomoNLP(blk)

        # set different default for `alpha_for_y` if this is an LP
        # see: https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_alpha_for_y
        if nlp is not None:
            if nlp.nnz_hessian_lag() == 0:
                if "alpha_for_y" not in self.options:
                    self.options["alpha_for_y"] = "bound-mult"

        kwds["options"] = self.options
        try:
            return solver.solve(blk, *args, **kwds)
        finally:
            self._cleanup(blk)

    def _cleanup_no_scale(self, blk):
        if self._dummy_objective_name is not None:
            blk.del_component(self._dummy_objective_name)
        self.options = self._original_options
        del self._original_options

    def _cleanup(self, blk):
        self._cleanup_no_scale(blk)


## reconfigure IDAES to use the ipopt-watertap solver
import idaes

_default_solver_config_value = idaes.cfg.get("default_solver")
_idaes_default_solver = _default_solver_config_value._default

_default_solver_config_value.set_default_value("ipopt-watertap")
if not _default_solver_config_value._userSet:
    _default_solver_config_value.reset()
