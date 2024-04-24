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
#
#################################################################################

__author__ = "Paul Vecchiarelli, Adam Atia"

import logging

from pyomo.environ import (
    value,
    units as pyunits,
    ConcreteModel,
    assert_optimal_termination,
)

from idaes.core import FlowsheetBlock, MaterialFlowBasis
from idaes.core.util.scaling import calculate_scaling_factors
from idaes.core.solvers import get_solver

from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
from watertap.tools.oli_api.util.chemistry_helper_functions import watertap_to_oli
from watertap.tools.oli_api.util.flash_helper_functions import input_unit_set

input_unit_set = input_unit_set

_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "OLIAPI - %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
)
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.DEBUG)

def create_state_block(
    inflows=None,
    inflows_unit=None,
    temperature=None,
    pressure=None,
):
    """
    Creates a state block using the Multi Component Aqueous Solution (MCAS) property model.

    :param inflows: dictionary containing species and concentrations
    :param inflows_unit: pyomo unit expression for inflows concentrations
    :param temperature: value for temperature (in K)
    :param pressure: value for pressure (in Pa)

    :return m: ConcreteModel containing MCAS state block
    """

    solver = get_solver()
    _logger.info("Creating model ...")
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    if inflows:
        props = create_property_model_input(inflows, property_model_type="mcas")
    else:
        raise RuntimeError(
            f"Inflows dictionary expected, not {inflows} {type(inflows)}"
        )
    m.fs.properties = MCASParameterBlock(
        **props, material_flow_basis=MaterialFlowBasis.mass
    )

    _logger.info("Creating state block ...")
    stream = m.fs.stream = m.fs.properties.build_state_block([0])
    if temperature is None:
        temperature = 298.15
    stream[0].temperature.fix(temperature)
    if pressure is None:
        pressure = 101325
    stream[0].pressure.fix(pressure)
    if inflows_unit is None:
        inflows_unit = pyunits.mg / pyunits.L
    stream[0].conc_mass_phase_comp
    _logger.info("Setting up model units and scaling ...")
    var_args = {}
    for k, v in inflows.items():
        val = pyunits.convert_value(
            v, inflows_unit, stream[0].conc_mass_phase_comp._units
        )
        var_args.update({("conc_mass_phase_comp", ("Liq", k)): val})
        if val == 0:
            scaling_factor = 1
        else:
            scaling_factor = 1/val
        m.fs.properties.set_default_scaling("conc_mass_phase_comp", scaling_factor, index=("Liq", k))
        m.fs.properties.set_default_scaling("flow_mass_phase_comp", scaling_factor, index=("Liq", k))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_vol_phase", 1, index=("Liq"))
    calculate_scaling_factors(m)
    var_args.update({("flow_vol_phase", "Liq"): 1e-3})
    stream.calculate_state(var_args=var_args, hold_state=True)
    stream.initialize()
    res = solver.solve(m)
    assert_optimal_termination(res)
    return m


def create_property_model_input(components, property_model_type: str = ""):
    """
    Build property package inputs.

    :param components: dictionary containing solute concentrations in mg/L
    :param property_model_type: string specifying property model to use

    :return property_model_inputs: dict containing inputs needed to build property model
    """

    if property_model_type == "mcas":
        solute_list = []
        mw_data = {}
        charge_data = {}

        for component in components:
            oli_comp = watertap_to_oli(component)
            solute_list.append(oli_comp.watertap_name)
            charge_data[component] = oli_comp.charge
            mw_data[component] = oli_comp.molar_mass

        property_model_inputs = {
            "solute_list": solute_list,
            "mw_data": mw_data,
            "charge": charge_data,
        }
        return property_model_inputs


def get_state_vars(state_block, inflow_name="flow_mass_phase_comp", flow_rate=None):
    """
    Extract state variables from WaterTAP unit model into a dictionary.

    :param state_block: input source to extract values from
    :param inflow_name: name of state_block component to extract inflows from
    :param flow_rate: pyomo unit expression for system flow rate

    :return oli_inflows: dictionary containing state variables
    """

    _logger.info(f"Extracting temperature and pressure from {state_block} ...")
    temp_component = state_block.find_component("temperature")
    if temp_component is not None:
        temperature = pyunits.convert_value(
            temp_component.value,
            temp_component.get_units(),
            input_unit_set["temperature"]["pyomo_unit"],
        )
        _logger.info("Extracted temperature")
    pres_component = state_block.find_component("pressure")
    if pres_component is not None:
        pressure = pyunits.convert_value(
            pres_component.value,
            pres_component.get_units(),
            input_unit_set["pressure"]["pyomo_unit"],
        )
        _logger.info("Extracted pressure")
    inflows = {}
    _logger.info(f"Extracting {inflow_name} from {state_block} ...")

    inflow_component = state_block.find_component(inflow_name)
    if inflow_component is not None:
        _logger.info(
            f"Reminder: WaterTAP inflows are in {inflow_component.get_units()}" +
            " and OLIApi requires units of mg/L.")
        raw_inflows = {k[-1]: v for k,v in inflow_component.items()}
        if flow_rate is not None:
            _logger.info(
                f"Converting inflows to {inflow_component.get_units()*flow_rate} ..."
                )
            inflows = {k: value(v*flow_rate) for k,v in raw_inflows.items()}
        else:
            inflows = {k: value(v) for k,v in raw_inflows.items()}
        _logger.info("Extracted inflows")

    oli_inflows = {
        "temperature": temperature,
        "pressure": pressure,
        "inflows": inflows,
    }
    return oli_inflows
