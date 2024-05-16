
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.collections import Bunch
from pyomo.contrib.fbbt.fbbt import fbbt

from watertap.core.plugins.solvers import IpoptWaterTAP


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
        self._value_cache = pyo.ComponentMap()

    def executable(self):
        return self._base_solver().executable()

    def _cache_bounds_values(self, blk):
        self._bound_cache.clear()
        self._value_cache.clear()
        for v in blk.component_data_objects(pyo.Var, active=True, descend_into=True):
            self._bound_cache[v] = v.bounds
            self._value_cache[v] = v.value

    def _restore_bounds(self):
        for v, bounds in self._bound_cache.items():
            v.bounds = bounds

    def _restore_values(self):
        for v, val in self._value_cache.items():
            v.set_value(val, skip_validation=True)

    def _fbbt(self, blk):
        # try:
        fbbt(
            blk,
            feasibility_tol=1e-6,
            deactivate_satisfied_constraints=False,
        )
        # except:
        #     # cleanup before raising
        #     self._restore_values()
        #     self._restore_bounds()
        #     raise

        for v in self._bound_cache:
            if v.value is None:
                # set to the bound nearer 0
                v.value = 0
            if v.lb is not None:
                if v.lb == v.ub:
                    v.value = v.lb
                    continue
                if v.value < v.lb:
                    # print(f"projecting {v.name} at value {v.value} onto lower bound {v.lb}")
                    v.set_value(v.lb, skip_validation=True)
            if v.ub is not None:
                if v.value > v.ub:
                    # print(f"projecting {v.name} at value {v.value} onto upper bound {v.ub}")
                    v.set_value(v.ub, skip_validation=True)

    def solve(self, blk, *args, **kwds):

        solver = self._base_solver()

        for k, v in self.options.items():
            solver.options[k] = v

        self._cache_bounds_values(blk)

        try:
            self._fbbt(blk)
        except InfeasibleConstraintException:
            # bounds / constraint restoration done
            # before exception is raised
            self._restore_values()

        self._restore_bounds()
        results = solver.solve(blk, *args, **kwds)

        return results
