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

import pytest
import pyomo.environ as pyo
import idaes.core.util.scaling as iscale

from idaes.core.util.scaling import (
    set_scaling_factor,
    constraints_with_scale_factor_generator,
)
from idaes.core.solvers import get_solver
from watertap.core.plugins.solvers import IpoptWaterTAP, _pyomo_nl_writer_log


class TestIpoptWaterTAP:
    @pytest.fixture(scope="class")
    def s(self):
        return pyo.SolverFactory("ipopt-watertap")

    @pytest.mark.unit
    def test_pyomo_registration(self, s):
        assert s.__class__ is IpoptWaterTAP

    @pytest.mark.unit
    def test_idaes_registration(self):
        assert get_solver().__class__ is IpoptWaterTAP

    @pytest.fixture(scope="class")
    def m2(self):
        m = pyo.ConcreteModel()
        m.factor = pyo.Param(initialize=1.0e-16, mutable=True)
        m.x = pyo.Var(bounds=(0.5 * m.factor, 1.5 * m.factor), initialize=m.factor)
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.x] = pyo.value(1.0 / m.factor)
        m.o = pyo.Objective(expr=m.x / m.factor)
        return m

    @pytest.mark.unit
    def test_default_bound_relax_small(self, m2, s):
        s.solve(m2, tee=True)
        assert pyo.value(m2.x) == pytest.approx(5.000000024092977e-17, abs=0, rel=1e-8)

    @pytest.mark.unit
    def test_default_bound_relax_big(self, m2, s):
        m2.factor = 1.0e16
        m2.x.value = 1.0e16
        m2.x.lb = 0.5 * m2.factor
        m2.x.ub = 1.5 * m2.factor
        m2.scaling_factor[m2.x] = pyo.value(1.0 / m2.factor)
        s.solve(m2, tee=True)
        assert pyo.value(m2.x) == pytest.approx(5.000000024092977e15, abs=0, rel=1e-8)
