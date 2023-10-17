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

import os

import pyomo.environ as pyo

from idaes.core import declare_process_block_class

from watertap.costing.zero_order_costing import ZeroOrderCostingData

_source_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "techno_economic",
    "watertap_costing_package.yaml",
)


@declare_process_block_class("WaterTAPCosting")
class WaterTAPCostingData(ZeroOrderCostingData):

    CONFIG = ZeroOrderCostingData.CONFIG()
    CONFIG.get("case_study_definition")._default = _source_file
    CONFIG.case_study_definition = _source_file

    def build_process_costs(self):
        super().build_process_costs()

        self.maintenance_labor_chemical_operating_cost = pyo.Expression(
            expr=(
                self.salary_cost
                + self.benefits_cost
                + self.maintenance_cost
                + self.laboratory_cost
                + self.insurance_and_taxes_cost
            )
        )
