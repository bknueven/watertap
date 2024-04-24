"""
This is a pretreatment train for brackish groundwater 5 (silica).

Core elements are:
    - BGW5 feed
    - softening with lime and soda
    - softening with magnesium oxide
    - acidification with HCl
    - brine concentration
"""

# imports
from pyomo.environ import (
    value,
    ConcreteModel,
    units as pyunits,
    TransformationFactory,
    check_optimal_termination
)
from pyomo.network import Arc

from idaes.models.unit_models import Feed, Product
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util.scaling import (
    calculate_scaling_factors,
)
from idaes.core.util.initialization import propagate_state

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    ActivityCoefficientModel,
    DensityCalculation,
    MaterialFlowBasis,
)
from watertap.unit_models.stoichiometric_reactor import StoichiometricReactor
from watertap.tools.oli_api.client import OLIApi
from watertap.tools.oli_api.credentials import CredentialManager
from watertap.tools.oli_api.flash import Flash
import watertap.tools.oli_api.util.chemistry_helper_functions as ch
import watertap.tools.oli_api.util.flash_helper_functions as fh
import watertap.tools.oli_api.util.state_block_helper_functions as sh
import watertap.tools.oli_api.silica_scaling_helper_functions as si_helpers

from watertap.tools.oli_api.util.postprocessing import scaling_tendency_plot

from numpy import linspace
from copy import deepcopy
import json


def build_model(inflows, flash_method, dbs_file_id, conc_unit, flow_rate, reactions):
    # run water analysis to get equilibrium concentrations from inflows
    inflows_eq = si_helpers.equilibrate_inflows(
        inflows=inflows,
        flash_method=flash_method,
        temperature=298.15,
        pressure=101325,
        reconciliation="EquilCalcOnly",
        ph=None,
        dbs_file_id=dbs_file_id,
        allow_solids=False,
    )
    if "[H2O]" in inflows_eq:
        water_conc = inflows_eq.pop("[H2O]")
        print(water_conc)

    # build property model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    props = sh.create_property_model_input(inflows_eq, property_model_type="mcas")
    props.update(
        {
            "activity_coefficient_model": ActivityCoefficientModel.ideal,
            "density_calculation": DensityCalculation.constant,
            "material_flow_basis": MaterialFlowBasis.mass,
        },
    )
    m.fs.properties = MCASParameterBlock(**props)

    # build unit models
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.feed.properties[0].temperature.fix()
    m.fs.feed.properties[0].pressure.fix()
    for comp, conc in inflows.items():
        mass_conc = conc*conc_unit
        mass_flow_rate = mass_conc*flow_rate
        m.fs.feed.properties[0].flow_mass_phase_comp[("Liq", comp)].fix(mass_flow_rate)
        if value(mass_flow_rate) == 0:
            scaling_factor = 1
        else:
            scaling_factor = 1/value(mass_flow_rate)
        m.fs.properties.set_default_scaling(
            "flow_mass_phase_comp",
            scaling_factor,
            index=("Liq", comp),
        )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp",
        1e-3,
        index=("Liq", "H2O"),
    )
    calculate_scaling_factors(m.fs.feed)
    m.fs.feed.initialize()
    reaction_stoichiometry = si_helpers.get_stoichiometry(reactions)
    # maximum lime dose will be determined by OLI survey
    m.fs.lime_reactor = StoichiometricReactor(
        property_package=m.fs.properties,
        reagent={"Ca[OH]2": reaction_stoichiometry["Ca[OH]2"]},
    )
    m.fs.lime_reactor.reagent_dose["Ca[OH]2"] #.fix(0*conc_unit)
    m.fs.s1 = Arc(source=m.fs.feed.outlet, destination=m.fs.lime_reactor.inlet)
    # maximum soda dose will be determined by OLI survey
    m.fs.soda_reactor = StoichiometricReactor(
        property_package=m.fs.properties,
        reagent={"Na2[CO3]": reaction_stoichiometry["Na2[CO3]"]},
    )
    m.fs.soda_reactor.reagent_dose["Na2[CO3]"] #.fix(0*conc_unit)
    m.fs.s2 = Arc(source=m.fs.lime_reactor.outlet, destination=m.fs.soda_reactor.inlet)
    # maximum mgox dose will be determined by OLI survey
    m.fs.mgox_reactor = StoichiometricReactor(
        property_package=m.fs.properties,
        reagent={"Mg[OH]2": reaction_stoichiometry["Mg[OH]2"]},
    )
    m.fs.mgox_reactor.reagent_dose["Mg[OH]2"] #.fix(0*conc_unit)
    m.fs.s3 = Arc(source=m.fs.soda_reactor.outlet, destination=m.fs.mgox_reactor.inlet)
    # acid dose will be determined by reconciliation to pH 7 in Product
    # % recovery will be determined by OLI survey (concentrating solutes in Product)
    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.s4 = Arc(source=m.fs.mgox_reactor.outlet, destination=m.fs.product.inlet)
    # initialize unit models and connections
    TransformationFactory("network.expand_arcs").apply_to(m)
    propagate_state(m.fs.s1)
    m.fs.lime_reactor.initialize()
    propagate_state(m.fs.s2)
    m.fs.soda_reactor.initialize()
    propagate_state(m.fs.s3)
    m.fs.mgox_reactor.initialize()
    propagate_state(m.fs.s4)
    m.fs.product.initialize()
    return m


def main():
    # set flash method for OLI calculations
    flash_method = "wateranalysis"
    try:
        # create DBS file for entire analysis
        with OLIApi(CredentialManager(), interactive_mode=False) as oliapi:
            dbs_file_id = oliapi.generate_dbs_file(inflows, keep_file=True)
    except:
        dbs_file_id = None

    # equilibrate inflows (BGW5)
    # inflows added into the treatment train after Feed should have values of 0
    inflows = {
        "H2O": 1e6,
        "Na+": 1120,
        #"Na2CO3": 0, # max 2000
        "CO3-2": 0,
        "K+": 15,
        "Ca+2": 150,
        #"Ca(OH)2": 0, # max 2000
        "OH-": 0,
        "Mg+2": 33,
        #"Mg(OH)2": 0, # max 2000
        "Cl-": 1750,
        "SO4-2": 260,
        "HCO3-": 250,
        "SiO2": 30.5,
    }
    # update inflows missing from reaktoro database
    # inflows = handle_missing_inflows(inflows, db)
    conc_unit = pyunits.mg/pyunits.L
    # specify input flow rate
    flow_rate = 1000*pyunits.L/pyunits.s
    # specify reactions that will occur in reactors
    reactions = {
        "Ca[OH]2": ["Ca_2+", "[OH]_-"],
        "Na2[CO3]": ["Na_+", "[CO3]_2-"],
        "Mg[OH]2": ["Mg_2+", "[OH]_-"],
    }
    # build initial model
    m = build_model(inflows, flash_method, dbs_file_id, conc_unit, flow_rate, reactions)
    m.fs.feed.properties[0].display()

    # get survey limits
    # conduct bisection solve on each reactor and max concentration of single additives
    # so maximum scaling tendency is < 1 at high % water recovery

    '''
    # specify surveys to conduct
    surveys = {
        "Ca[OH]2": flash.build_survey(
            {"Ca[OH]2": linspace(0, 500, 10)}, get_oli_names=True,
        ),
        "Na2[CO3]": flash.build_survey(
            {"Na2[CO3]": linspace(0, 500, 10)}, get_oli_names=True,
        ),
        "Mg[OH]2": flash.build_survey(
            {"Mg[OH]2": linspace(0, 1e3, 10)}, get_oli_names=True,
        ),
        "recovery": flash.build_survey(
            {"recovery": linspace(40, 90, 10)}, get_oli_names=True,
        ),
    }
    '''

    # solve model with OLIApi
    solver = get_solver()
    results = solver.solve(m)
    check_optimal_termination(results)
    return results


if __name__ == "__main__":

    main()
