###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

from collections.abc import MutableMapping
import os

import pyomo.environ as pyo

from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core import declare_process_block_class
from idaes.core.base.costing_base import (
    FlowsheetCostingBlockData,
    register_idaes_currency_units,
)

from idaes.models.unit_models import Mixer

from watertap.core.zero_order_costing import ZeroOrderCostingData
from watertap.unit_models import (
    ReverseOsmosis0D,
    ReverseOsmosis1D,
    NanoFiltration0D,
    NanofiltrationZO,
    PressureExchanger,
    Crystallization,
    Ultraviolet0D,
    Pump,
    EnergyRecoveryDevice,
    Electrodialysis0D,
    Electrodialysis1D,
    IonExchange0D,
    GAC,
)

from .units.crystallizer import cost_crystallizer
from .units.electrodialysis import cost_electrodialysis
from .units.energy_recovery_device import cost_energy_recovery_device
from .units.gac import cost_gac
from .units.ion_exchange import cost_ion_exchange
from .units.nanofiltration import cost_nanofiltration
from .units.mixer import cost_mixer
from .units.pressure_exchanger import cost_pressure_exchanger
from .units.pump import cost_pump
from .units.reverse_osmosis import cost_reverse_osmosis
from .units.uv_aop import cost_uv_aop


class _DefinedFlowsDict(MutableMapping, dict):
    # use dict methods
    __getitem__ = dict.__getitem__
    __iter__ = dict.__iter__
    __len__ = dict.__len__

    def __setitem__(self, key, value):
        if key in self and self[key] is not value:
            raise KeyError(f"{key} has already been defined as a flow")
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        raise KeyError("defined flows cannot be removed")


_source_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "techno_economic",
    "watertap_costing_package.yaml",
)

@declare_process_block_class("WaterTAPCosting")
class WaterTAPCostingData(ZeroOrderCostingData):
    # Define default mapping of costing methods to unit models
    unit_mapping = {**ZeroOrderCostingData.unit_mapping}
    unit_mapping.update({
        Mixer: cost_mixer,
        Pump: cost_pump,
        EnergyRecoveryDevice: cost_energy_recovery_device,
        PressureExchanger: cost_pressure_exchanger,
        ReverseOsmosis0D: cost_reverse_osmosis,
        ReverseOsmosis1D: cost_reverse_osmosis,
        NanoFiltration0D: cost_nanofiltration,
        NanofiltrationZO: cost_nanofiltration,
        Crystallization: cost_crystallizer,
        Ultraviolet0D: cost_uv_aop,
        Electrodialysis0D: cost_electrodialysis,
        Electrodialysis1D: cost_electrodialysis,
        IonExchange0D: cost_ion_exchange,
        GAC: cost_gac,
    })

    CONFIG = ZeroOrderCostingData.CONFIG()
    CONFIG.get("case_study_definition")._default = _source_file
    CONFIG.case_study_definition = _source_file

    def build_global_params(self):
        super().build_global_params()

        # Can ADD
        self.electrical_carbon_intensity = pyo.Param(
            mutable=True,
            initialize=0.475,
            doc="Grid carbon intensity [kgCO2_eq/kWh]",
            units=pyo.units.kg / pyo.units.kWh,
        )

        # fix the parameters
        self.fix_all_vars()

    def cost_flow(self, flow_expr, flow_type):
        """
        This method registers a given flow component (Var or expression) for
        costing. All flows are required to be bounded to be non-negative (i.e.
        a lower bound equal to or greater than 0).

        Args:
            flow_expr: Pyomo Var or expression that represents a material flow
                that should be included in the process costing. Units are
                expected to be on a per time basis.
            flow_type: string identifying the material this flow represents.
                This string must be available to the FlowsheetCostingBlock
                as a known flow type.

        Raises:
            ValueError if flow_type is not recognized.
            TypeError if flow_expr is an indexed Var.
        """
        if flow_type not in self.defined_flows:
            raise ValueError(
                f"{flow_type} is not a recognized flow type. Please check "
                "your spelling and that the flow type has been available to"
                " the FlowsheetCostingBlock."
            )
        if flow_type not in self.flow_types:
            self.register_flow_type(flow_type, self.defined_flows[flow_type])
        super().cost_flow(flow_expr, flow_type)

    def build_process_costs(self):
        super().build_process_costs()

        self.total_investment_cost = pyo.Expression(
            expr=self.total_capital_cost
        )
        self.maintenance_labor_chemical_operating_cost = pyo.Expression(
            expr=(self.salary_cost
            + self.benefits_cost
            + self.maintenance_cost
            + self.laboratory_cost
            + self.insurance_and_taxes_cost)
            )


    def add_LCOW(self, flow_rate, name="LCOW"):
        """
        Add Levelized Cost of Water (LCOW) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating LCOW
            name (optional) - name for the LCOW variable (default: LCOW)
        """

        LCOW = pyo.Var(
            doc=f"Levelized Cost of Water based on flow {flow_rate.name}",
            units=self.base_currency / pyo.units.m**3,
        )
        self.add_component(name, LCOW)

        LCOW_constraint = pyo.Constraint(
            expr=LCOW
            == (
                self.total_investment_cost * self.factor_capital_annualization
                + self.total_operating_cost
            )
            / (
                pyo.units.convert(
                    flow_rate, to_units=pyo.units.m**3 / self.base_period
                )
                * self.utilization_factor
            ),
            doc=f"Constraint for Levelized Cost of Water based on flow {flow_rate.name}",
        )
        self.add_component(name + "_constraint", LCOW_constraint)

        self._registered_LCOWs[name] = (LCOW, LCOW_constraint)

    def add_annual_water_production(self, flow_rate, name="annual_water_production"):
        """
        Add annual water production to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating annual water production
            name (optional) - name for the annual water productionvariable
                              Expression (default: annual_water_production)
        """
        self.add_component(
            name,
            pyo.Expression(
                expr=(
                    pyo.units.convert(
                        flow_rate, to_units=pyo.units.m**3 / self.base_period
                    )
                    * self.utilization_factor
                ),
                doc=f"Annual water production based on flow {flow_rate.name}",
            ),
        )

    def add_specific_energy_consumption(
        self, flow_rate, name="specific_energy_consumption"
    ):
        """
        Add specific energy consumption (kWh/m**3) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating specific energy consumption
            name (optional) - the name of the Expression for the specific
                              energy consumption (default: specific_energy_consumption)
        """

        self.add_component(
            name,
            pyo.Expression(
                expr=self.aggregate_flow_electricity
                / pyo.units.convert(
                    flow_rate, to_units=pyo.units.m**3 / pyo.units.hr
                ),
                doc=f"Specific energy consumption based on flow {flow_rate.name}",
            ),
        )

    def add_specific_electrical_carbon_intensity(
        self, flow_rate, name="specific_electrical_carbon_intensity"
    ):
        """
        Add specific electrical carbon intensity (kg_CO2eq/m**3) to costing block.
        Args:
            flow_rate - flow rate of water (volumetric) to be used in
                        calculating specific electrical carbon intensity
            name (optional) - the name of the Expression for the specific
                              energy consumption (default: specific_electrical_carbon_intensity)
        """

        self.add_component(
            name,
            pyo.Expression(
                expr=self.aggregate_flow_electricity
                * self.electrical_carbon_intensity
                / pyo.units.convert(
                    flow_rate, to_units=pyo.units.m**3 / pyo.units.hr
                ),
                doc=f"Specific electrical carbon intensity based on flow {flow_rate.name}",
            ),
        )
