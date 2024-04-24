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

import reaktoro as rkt

import silica_scaling_helper_functions as hf

from numpy import linspace
from copy import deepcopy
import json

'''
# this will be used to optimize water recovery in Product, and to find maximum values for single additives in the Reactors
def _bisection_solve(x_max, y_max=1, tolerance=0.1, x_min=0, iterations=20):
    " Find minimum dosage required to minimize scaling in product. "
    y_test = _get_scaling_tendencies()
    for i in range(iterations):
        x_test = (x_max - x_min)/2
        # compare test value for dependent variable with max allowed value
        y_test = _get_scaling_tendencies()
        if y_test > y_max:
            x_max = x_test
        else:
            if (y_max - tolerance) <= y_test <= y_max:
                return x_test
            else:
                x_min = x_test
    return None

def find_max_dosages(m, reactors):
    for additive, data in reactors.items():
        print(data["reactor"].reagent_dose[additive])
'''

def build_model(db, inflows, flow_rate, conc_unit, temp, pres, reactions):
    " Build WaterTAP flowsheet. "

    # build flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    components = hf.find_components(inflows, db)
    props = {
        "solute_list": [i for i in inflows if i != "H2O"],
        "mw_data": hf.get_molar_mass(components),
        "charge": hf.get_charge(components),
        "activity_coefficient_model": ActivityCoefficientModel.ideal,
        "density_calculation": DensityCalculation.constant,
        "material_flow_basis": MaterialFlowBasis.mass,
    }
    m.fs.properties = MCASParameterBlock(**props)

    # set up unit models
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.lime_reactor = StoichiometricReactor(
        property_package=m.fs.properties,
        reagent={"lime": reactions["lime"]},
    )
    m.fs.lime_reactor.reagent_dose["lime"].fix(0 * conc_unit)
    m.fs.soda_reactor = StoichiometricReactor(
        property_package=m.fs.properties,
        reagent={"soda": reactions["soda"]},
    )
    m.fs.soda_reactor.reagent_dose["soda"].fix(0 * conc_unit)
    m.fs.mgox_reactor = StoichiometricReactor(
        property_package=m.fs.properties,
        reagent={"mgox": reactions["mgox"]},
    )
    m.fs.mgox_reactor.reagent_dose["mgox"].fix(0 * conc_unit)
    m.fs.product = Product(property_package=m.fs.properties)

    # build and expand arcs
    m.fs.s1 = Arc(source=m.fs.feed.outlet, destination=m.fs.lime_reactor.inlet)
    m.fs.s2 = Arc(source=m.fs.lime_reactor.outlet, destination=m.fs.soda_reactor.inlet)
    m.fs.s3 = Arc(source=m.fs.soda_reactor.outlet, destination=m.fs.mgox_reactor.inlet)
    m.fs.s4 = Arc(source=m.fs.mgox_reactor.outlet, destination=m.fs.product.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)
    return m

def main():
    solver = get_solver()

    # state variables
    #conc_unit = 1 * (pyunits.mg/pyunits.L)
    #flow_rate = 1000 * (pyunits.L/pyunits.s)
    temp = 298.15 * pyunits.K
    pres = 101325 * pyunits.Pa

    # inflows (BGW5) in mg/L
    inflows = {
        "H2O": 1e6,
        "H+": 0,
        #"Na+": 1120,
        "CO3-2": 250,
        #"K+": 15,
        "Ca+2": 150,
        "OH-": 0,
        #"Mg+2": 33,
        #"Cl-": 1750,
        #"SO4-2": 260,
        #"HCO3-": 250,
        #"SiO2(aq)": 30.5,
    }

    # create a system in Reaktoro
    db = rkt.PhreeqcDatabase("pitzer.dat")
    #feed_system = hf.create_chemical_system(db, inflows, allow_solids=False)
    #feed_solver, feed_specs = hf.create_equilibrium_specs(feed_system, ph=True)
    # create initial state to equilibrate feed without forming solids
    #feed_state, feed_aq_props, feed_chem_props = hf.create_state(
    #    feed_system, inflows, temp, pres,
    #)

    # build initial model
    #additives = {"lime": 1000, "soda": 0, "mgox": 0} # dose in mg/L
    #reactions = hf.get_stoich_reactor_input(additives)
    #m = build_model(db, inflows, flow_rate, conc_unit, temp, pres, reactions)

    #print("Feed Unit: ")
    # equilibrate feed to pH specification, but do not form any solids
    #print("Reaktoro: equilibrating feed to pH 7.3 with no solid formation")
    #hf.equilibrate_state(feed_state, feed_aq_props, feed_chem_props, feed_solver, feed_specs, ph=7.3)
    #feed_stream = hf.get_aqueous_stream(feed_system, feed_state, feed_aq_props, feed_chem_props)
    #print(feed_specs.titrants())
    #print(feed_state.equilibrium().namesInputVariables())
    #print(*(species.name() for species in feed_system.species()))

    # update and initialize feed unit model based on equilibrated feed
    #blk = m.fs.feed.properties[0]
    #hf.update_unit(blk, feed_stream, flow_rate, conc_unit, arc=m.fs.s1)
    #blk.temperature.fix(temp)
    #blk.pressure.fix(pres)
    #for comp, conc in blk.flow_mass_phase_comp.items():
    #    scaling_factor = 1/value(conc)
    #    m.fs.properties.set_default_scaling(
    #        "flow_mass_phase_comp", scaling_factor, index=("Liq", comp),
    #    )
    #calculate_scaling_factors(blk.flowsheet().properties)
    #blk.flow_mass_phase_comp.display()
    # create a new system in Reaktoro to account for solid formation and pass on to lime reactor
    reactor_system = hf.create_chemical_system(db, inflows, allow_solids=True)
    reactor_solver, reactor_specs = hf.create_equilibrium_specs(reactor_system, ph=False, additive="Ca(OH)2")
    #reactor_state, reactor_aq_props, reactor_chem_props = hf.create_state(
    #    reactor_system, inflows, temp, pres,
    #)
    print("Lime reactor: ")

    # get flow rates from lime reactor after amendment and equilibration
    #blk = m.fs.lime_reactor.dissolution_reactor.properties_out[0]
    #hf.amend_system(
    #    reactor_system, reactor_state, additives["lime"],
    #    reactor_solver, reactor_specs, reactions["lime"],
    #)
    #hf.equilibrate_state(
    #    reactor_state, reactor_aq_props, reactor_chem_props, reactor_solver, reactor_specs, ph=None,
    #)
    #lime_stream = hf.get_aqueous_stream(
    #    reactor_system, reactor_state, reactor_aq_props, reactor_chem_props,
    #)
    #print({k:value(v) for k,v in lime_stream.items()})
    print(reactor_specs.inputs())
    #print(feed_state.equilibrium().namesInputVariables())
    #print(*(species.name() for species in feed_system.species()))
    #hf.update_unit(blk, lime_stream, flow_rate, conc_unit, arc=m.fs.s2)
    #blk.flow_mass_phase_comp.display()

    '''
    # get component flow rates from soda reactor after amendment and equilibration
    print("Soda Ash reactor: ")
    #blk = m.fs.soda_reactor.dissolution_reactor.properties_out[0]
    hf.amend_system(
        reactor_system, lime_state, additives["soda"],
        reactor_solver, reactor_specs, reactions["soda"],
    )
    hf.equilibrate_state(
        lime_state, lime_aq_props, lime_chem_props, reactor_solver, reactor_specs, ph=None,
    )
    soda_stream = hf.get_aqueous_stream(
        reactor_system, lime_state, lime_aq_props, lime_chem_props,
    )
    print({k:value(v) for k,v in soda_stream.items()})

    #hf.update_unit(blk, soda_stream, flow_rate, conc_unit, arc=m.fs.s3)
    #blk.flow_mass_phase_comp.display()

    # create new state from aqueous stream and pass on to mgox reactor
    soda_state, soda_aq_props, soda_chem_props = hf.create_state(
        reactor_system, soda_stream, flow_rate, conc_unit, temp, pres,
    )

    print("Magnesium Oxide reactor: ")
    # get component flow rates from mgox reactor after amendment and equilibration
    #blk = m.fs.mgox_reactor.dissolution_reactor.properties_out[0]
    hf.amend_system(
        reactor_system, soda_state, additives["mgox"],
        reactor_solver, reactor_specs, reactions["mgox"],
    )
    hf.equilibrate_state(
        soda_state, soda_aq_props, soda_chem_props, reactor_solver, reactor_specs, ph=None,
    )
    mgox_stream = hf.get_aqueous_stream(
        reactor_system, reactor_state, reactor_aq_props, reactor_chem_props,
    )
    print({k:value(v) for k,v in mgox_stream.items()})
    #hf.update_unit(blk, mgox_stream, flow_rate, conc_unit, arc=m.fs.s4)
    #blk.flow_mass_phase_comp.display()

    # create new state from aqueous stream and pass on to product
    mgox_state, mgox_aq_props, mgox_chem_props = hf.create_state(
        reactor_system, mgox_stream, flow_rate, conc_unit, temp, pres,
    )

    print("Product Unit: ")
    # equilibrate product to pH specification and form solids
    product_solver, product_specs = hf.create_equilibrium_specs(reactor_system, ph=True)
    print("Reaktoro: equilibrating product to pH 7.0 with solid formation")

    hf.equilibrate_state(
        mgox_state, mgox_aq_props, mgox_chem_props, product_solver, product_specs, ph=7.0,
    )
    product_stream = hf.get_aqueous_stream(
        reactor_system, mgox_state, mgox_aq_props, mgox_chem_props,
    )
    print({k:value(v) for k,v in product_stream.items()})

    #blk = m.fs.product.properties[0]
    #hf.update_unit(blk, product_stream, flow_rate, conc_unit)
    #print(f"Amount of acid added: {reactor_state.equilibrium().titrantAmount('H+')} mol")
    #blk.flow_mass_phase_comp.display()

    # concentrate product stream
    prod_state, prod_aq_props, prod_chem_props = hf.create_state(
        reactor_system, product_stream, flow_rate, conc_unit, temp, pres,
    )

    #h2o_init = product_stream["H2O"]
    #conc_streams = {}
    #for pct_recovery in [10, 50, 80]:
    #    h2o_conc = hf.water_recovery(h2o_init, pct_recovery)
    #    print(f"Reaktoro: concentrating product to {pct_recovery} % recovery")
    #    prod_state.set("H2O", h2o_conc, "mg")
    #    conc_stream = hf.get_aqueous_stream(
    #        reactor_system, prod_state, prod_aq_props, prod_chem_props,
    #    )
    #    conc_streams[pct_recovery] = conc_stream
    '''
if __name__ == "__main__":
    main()
