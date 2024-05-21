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

from numpy import linspace
from copy import deepcopy
import json

import matplotlib.pyplot as pt

def create_chemical_system(db, inflows, allow_solids=False):
    " Create a chemical system. "
    #inflows = {(k if k != "H2O" else "H2O(aq)"):v for k,v in inflows.items()}
    aq_phase = rkt.AqueousPhase([k for k in inflows])
    aq_phase.set(rkt.ActivityModelPitzer())
    construction_args = [db, aq_phase]
    if allow_solids:
        construction_args.append(rkt.MineralPhases())
    # TODO: setup kinetics
    system = rkt.ChemicalSystem(*construction_args)
    return system

def create_equilibrium_specs(system, ph=False):
    " Create solver for given equilibrium specification. "
    specs = rkt.EquilibriumSpecs(system)
    specs.temperature()
    specs.pressure()
    if ph:
        specs.pH()
    solver = rkt.EquilibriumSolver(specs)
    return solver, specs

def create_state(system, inflows, temp, pres):
    " Create an initial state in Reaktoro. "
    state = rkt.ChemicalState(system)
    state.temperature(value(temp), str(pyunits.get_units(temp)))
    state.pressure(value(pres), str(pyunits.get_units(pres)))
    #inflows = {(k if k != "H2O" else "H2O(aq)"):v for k,v in inflows.items()}
    for inflow, conc in inflows.items():
        try:
            conc = value(conc)
        except:
            pass
        if conc == 0:
            state.set(inflow, 1e-16, "mg")
        else:
            state.set(inflow, conc, "mg")
    #state.scaleVolume(1, "L")
    aq_props = rkt.AqueousProps(state)
    chem_props = rkt.ChemicalProps(state)
    return state, aq_props, chem_props

def find_components(comp_names, db):
    " Find components in db. "
    components = {}
    names_db = {species.name(): species for species in db.species()}
    for name in comp_names:
        #if name != "H2O":
        if name in names_db:
            components[name] = names_db[name]
        else:
            print(f"{name} not in db")
        #else:
        #    components["H2O"] = names_db["H2O(aq)"]
    return components

def get_molar_mass(components):
    " Return molar mass value (g/mol). "
    molar_mass_values = {}
    for component, data in components.items():
        molar_mass_kg = data.molarMass() * (pyunits.kg/pyunits.mol)
        molar_mass_values[component] = pyunits.convert(molar_mass_kg, to_units=pyunits.g/pyunits.mol)
    return molar_mass_values

def get_charge(components):
    " Return charge value (unitless). "
    charge_values = {}
    for component, data in components.items():
        if component != "H2O":
            charge_values[component] = data.charge()
    return charge_values

def get_stoich_reactor_input(reactions):
    " Manually specify aqueous dissolution reactions. "
    reactions = {
        "lime": {
            "mw": 74.093 * (pyunits.g / pyunits.mol),
            "dissolution_stoichiometric": {"Ca+2": 1, "OH-": 2},
            "density_reagent": 2.21 * (pyunits.kg / pyunits.L),
        },
        "soda": {
            "mw": 105.9888 * (pyunits.g / pyunits.mol),
            "dissolution_stoichiometric": {"Na+": 2, "CO3-2": 1},
            "density_reagent": 2.54 * (pyunits.kg / pyunits.L),
        },
        "mgox": {
            "mw": 58.3197 * (pyunits.g / pyunits.mol),
            "dissolution_stoichiometric": {"Mg+2": 1, "OH-": 2},
            "density_reagent": 2.34 * (pyunits.kg / pyunits.L),
        }
    }
    return reactions


# functions above this line have been checked
########################################################################################

def equilibrate_state(state, aq_props, chem_props, solver, specs, ph=None):
    " Equilibrate a Reaktoro state and update properties. "
    conditions = rkt.EquilibriumConditions(specs)
    sensitivity = rkt.EquilibriumSensitivity(specs)
    if ph:
        conditions.pH(ph)
    result = solver.solve(state, sensitivity, conditions)
    assert result.succeeded()
    aq_props.update(state)
    chem_props.update(state)
    return

def get_aqueous_stream(system, state, aq_props, chem_props):
    " Get aqueous solutes from equilibrated stream in mg/L. "
    aq_stream = {}
    system_density = pyunits.convert(
        chem_props.density().val() * (pyunits.kg / pyunits.m**3),
        to_units=(pyunits.kg / pyunits.L),
    )
    for idx, molality in enumerate(aq_props.speciesMolalities()):
        species = system.species(idx)
        #if species.name() == "H2O(aq)":
        #    name = "H2O"
        #else:
        name = species.name()
        species_mw = species.molarMass() * (pyunits.kg / pyunits.mol)
        species_molal = molality.val() * (pyunits.mol / pyunits.kg)
        aq_stream[name] = pyunits.convert(
            species_molal * system_density * species_mw,
            to_units=(pyunits.mg / pyunits.L),
        )
    return aq_stream

def amend_system(system, state, additive, solver, specs, reaction):
    " Add an additive to a Reaktoro/WaterTAP system. "
    product_mw = {}
    for product in reaction["dissolution_stoichiometric"]:
        for idx, species in enumerate(system.species()):
            if product == species.name():
                product_mw[product] = pyunits.convert(
                    system.species(idx).molarMass() * (pyunits.kg / pyunits.mol),
                    to_units=pyunits.g/pyunits.mol,
                )
    for product, coefficient in reaction["dissolution_stoichiometric"].items():
        moles_product = coefficient * ((additive * pyunits.mg) / reaction["mw"])
        mass_product = pyunits.convert(moles_product * product_mw[product], to_units=pyunits.mg)
        print(f"Reaktoro: adding {value(mass_product)} {pyunits.get_units(mass_product)} of {product}")
        state.set(product, value(mass_product), str(pyunits.get_units(mass_product)))
    return

def remove_solids(system, state, aq_props, chem_props):
    scalants = get_scaling_tendencies(aq_props)
    print(scalants)

def get_scaling_tendencies(aq_props):
    " Determine scaling tendency for system components. "
    potential_scalants = aq_props.saturationSpecies()
    scalants = {}
    for scalant in potential_scalants:
        name = scalant.name()
        if aq_props.saturationIndex(scalant.name()) > 0.0:
            scalants[name] = scalant
    return scalants

def update_unit(blk, aq_stream, flow_rate, conc_unit, arc=None):
    " Update equilibrium concentrations of unit models. "
    parent_blk = blk.parent_block()
    if "properties_out" in dir(parent_blk):
        parent_blk = parent_blk.parent_block()
    parent_blk.initialize()
    # amend stream with reactants and update
    for comp, conc in blk.flow_mass_phase_comp.items():
        conc.fix(aq_stream[comp[-1]] * conc_unit * flow_rate)
    if arc is not None:
        propagate_state(arc)

def extract_inflows(blk, flow_rate, conc_unit, inflow_name="flow_mass_phase_comp"):
    " Extract concentrations from WaterTAP state block. "
    inflows = {}
    inflow_component = blk.find_component(inflow_name)
    if inflow_component is not None:
        #print(f"Converting {inflow_name} from {inflow_component.get_units()} to {conc_unit}.")
        for k, v in inflow_component.items():
            if "flow" in inflow_name:
                v = v / flow_rate
            v = pyunits.convert(v, to_units=conc_unit)
            inflows[k[-1]] = v
    return inflows

def water_recovery(h2o_init, pct_recovery):
    conc_factor = 1 - (pct_recovery / 100)
    h2o_final = h2o_init * conc_factor
    return h2o_final

# Needed with Phreeqc but not SupCrt
def _replace_si_species(inflows, db):
    '''
    # check if all species are in the db
    species = [species.name() for species in db.species()]
    for inflow in inflows:
        if inflow not in species:
            print(f"{inflow} not in db")
    # aqueous silica is not in the PHREEQC db
    # find alternative aqueous component containing silica
    for species in db.species():
        if "Si" in str(species.formula()):
            print(species.formula(), species.molarMass())
    '''
    # convert SiO2 to H4SiO4
    sio2_mass_conc = inflows.pop("SiO2") / 1000 # g SiO2 / L
    sio2_molar_mass = 0.0600843 * 1000 # g SiO2 / mol SiO2
    sio2_molarity = sio2_mass_conc / sio2_molar_mass # mol SiO2 / L
    sio2_stoich = 1 # mol Si / mol SiO2
    si_molarity = sio2_molarity * sio2_stoich # mol Si / L
    h4sio4_stoich = 1 # mol H4SiO4 / mol Si
    h4sio4_molarity = h4sio4_stoich * si_molarity # mol H4SiO4 / L
    h4sio4_molar_mass = 0.0961163 * 1000 # g H4SiO4 / mol H4SiO4
    h4sio4_mass_conc = h4sio4_molarity * h4sio4_molar_mass * 1000 # mg H4SiO4 / L
    inflows["H4SiO4"] = h4sio4_mass_conc
    return inflows

inflows = {
    "H2O": 1e6,
    "H+": 0,
    "Na+": 1120,
    "CO3-2": 0,
    "K+": 15,
    "Ca+2": 150,
    "OH-": 0,
    "Mg+2": 33,
    "Cl-": 1750,
    "SO4-2": 260,
    "HCO3-": 250,
    "SiO2": 30.5,
}

_replace_si_species(inflows, db = rkt.PhreeqcDatabase("pitzer.dat"))
