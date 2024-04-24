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

"""
This file contains methods to convert WaterTAP naming conventions to OLI
and generate molecular weight and charge dictionaries from molecular formulae.

It calculates molecular weights using the periodic_table.csv from:
https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee.
"""

__author__ = "Paul Vecchiarelli, Ben Knueven, Adam Atia"

from collections import namedtuple

import re

from pathlib import Path
from os.path import join
from pandas import read_csv
from pyomo.environ import units as pyunits
# used in stoichiometry function only
import numpy as np
from sympy import Matrix, shape
from fractions import Fraction
from functools import reduce
from math import gcd


OLIName = namedtuple(
    "OLIName", ["oli_name", "watertap_name", "charge", "charge_group", "molar_mass"]
)


def watertap_to_oli(watertap_name: str) -> OLIName:
    """
    Create a named tuple which can be passed directly into OLI or into MCAS property models.

    :param watertap_name: string name of substance in WaterTAP format, i.e., B[OH]4_-

    :return OLIName: named tuple containing attributes derived from molecular formula
    """

    c = re.findall(r"[A-Z]", watertap_name)
    if len(c) == 0:
        raise IOError(
            f" At least 1 uppercase letter is required to specify a molecule, not '{watertap_name}'."
        )

    # check if given name is an existing OLI name
    if watertap_name in names_db:
        watertap_name = names_db[watertap_name]
    # get attributes and store in namedtuple object
    oli_name = get_oli_name(watertap_name)
    charge = get_charge(watertap_name)
    charge_group = get_charge_group(charge)
    molar_mass = get_molar_mass_quantity(watertap_name)
    return OLIName(oli_name, watertap_name, charge, charge_group, molar_mass)


def get_oli_name(watertap_name: str) -> str:
    """
    Convert an WaterTAP formatted name, i.e., "Na_+" into an OLI formatted name, i.e., "NAION".

    :param watertap_name: string name of a solute in WaterTAP format

    :return oli_name: string name of a solute in OLI format
    """

    # exclude names of state variables that could be passed in from other functions
    exclude_items = ["temperature", "pressure", "volume", "recovery"]
    if watertap_name.lower() in exclude_items:
        return watertap_name
    if hasattr(watertap_name, "oli_name"):
        return watertap_name
    components = watertap_name.split("_")
    if len(components) == 0:
        raise IOError(f" Unable to parse solute '{watertap_name}'.")
    if len(components) == 1:
        molecule = components[0]
    elif len(components) == 2:
        molecule = components[0] + "ION"
    oli_name = molecule.replace("[", "").replace("]", "").upper()
    return oli_name


def get_charge(watertap_name: str) -> int:
    """
    Get charge from WaterTAP formatted names.

    :param watertap_name: string name of a solute in WaterTAP format

    :return charge: integer value of charge
    """

    components = watertap_name.split("_")
    if len(components) == 0:
        raise IOError(f" Unable to parse solute '{watertap_name}'.")
    if len(components) == 1:
        molecule = components[0]
        charge = 0
    elif len(components) == 2:
        molecule = components[0] + "ION"
        charge = components[1]
        try:
            charge_sign = charge[-1]
        except IndexError:
            raise IOError(
                f"Charge sign could not be determined from the string '{watertap_name}'"
            )
        if len(charge) > 1:
            try:
                charge_magnitude = int(charge[:-1])
            except ValueError:
                raise IOError(
                    f"Charge sign could not be determined from the string '{watertap_name}'"
                )
        else:
            charge_magnitude = 1
        if charge_sign == "+":
            charge = charge_magnitude
        elif charge_sign == "-":
            charge = -charge_magnitude
        else:
            raise IOError(
                f"Only + and - are valid charge indicators and neither was provided in '{watertap_name}'."
            )
    else:
        raise IOError(
            f"Charge could not be determined from the string '{watertap_name}'"
        )
    return charge


def get_charge_group(charge: int) -> str:
    """
    Categorize molecule based on its charge.

    :param charge: integer value for charge

    :return group: string name for charge group
    """

    if charge == 0:
        group = "Neutrals"
    elif charge > 0:
        group = "Cations"
    elif charge < 0:
        group = "Anions"
    return group


def get_stoichiometric_expression(watertap_name: str, products: list) -> dict:
    """
    Determine stoichiometry of a dissolution or precipitation reaction based on a neutral component (watertap_name) and set of ions (products).

    :param watertap_name: string name of a solute in WaterTAP format
    :param products: list containing names of ionic solutes in WaterTAP format

    :return stoichiometry: dictionary mapping species involved in reaction and their stoichiometric coefficients
    """

    system = []
    stoichiometry = {}
    null_atom_counts = {}
    system_reactants = []
    reactant_atom_counts = get_atom_counts(watertap_name)
    for k,v in reactant_atom_counts.items():
        null_atom_counts[k] = 0
        system_reactants.append(v)
    system.append(system_reactants)
    for product in products:
        product_atom_counts = {
            **null_atom_counts, **get_atom_counts(product.split("_")[0]),
        }
        system.append([-v for v in product_atom_counts.values()])
    # get rref matrix and pivots
    m, p = Matrix(np.array(system).transpose()).rref()
    n_row, n_col = shape(m)
    if n_col > n_row:
        m = m.row_insert(n_row, Matrix([[0]*n_col]))
    denominators = [Fraction(v).limit_denominator().denominator for v in -m.col(-1)]
    lcm = reduce(lambda x,y: x*y//gcd(x,y), denominators)
    # determine stoichiometric coefficients
    species = [watertap_name, *products]
    for idx in range(shape(m)[1]):
        if m[idx,-1] == 0:
            val = 1
        else:
            val = -m[idx,-1]
        stoichiometry[species[idx]] = lcm * val
    # TODO: test product vs. reactant sums
    return stoichiometry


def get_atom_counts(molecule: str) -> dict:
    """
    Get atoms and quantities for a given molecule.

    :param molecule: string name of a molecule in WaterTAP format

    :return atom_counts: dictionary mapping atom names and molar quantities
    """

    def _get_groups(string):
        initial_bound = None
        groups = []
        lower_bounds = [idx for idx,c in enumerate(string) if c == "["]
        upper_bounds = [idx for idx,c in enumerate(string) if c == "]"]
        if lower_bounds:
            initial_bound = lower_bounds[0]
        groups.append(string[0:initial_bound])
        for idx in range(len(lower_bounds)):
            ub_offset = 1
            ub = upper_bounds[idx]
            lb = lower_bounds[idx]
            try:
                if string[ub+ub_offset].isnumeric():
                    ub_offset = 2
            except IndexError:
                pass
            groups.append(string[lb:ub+ub_offset])
        return groups

    def _get_atoms(string):
        atoms = {}
        group_coefficient = 1
        group_end = string.find("]")
        if group_end != -1:
            try:
                if string[group_end+1].isnumeric:
                    group_coefficient = int(string[group_end+1])
            except IndexError:
                pass
        for atom in re.findall("[A-Z][a-z]?[0-9]*", string):
            name, coefficient = _get_atom_count(atom)
            atoms[name] = coefficient * group_coefficient
        return atoms

    def _get_atom_count(atom: str) -> dict:
        # single- or double- letter atom with assumed 1 stoichiometry
        if len(atom) == 1 or (len(atom) == 2 and atom.isalpha()):
            symbol = atom
            coefficient = 1
        # single- or double- letter atom with single digit stoichiometry
        elif (len(atom) == 2 and not atom.isalpha()) or (len(atom) == 3 and atom[:-1].isalpha()):
            symbol = atom[:-1]
            coefficient = int(atom[-1])
        # double letter element with double digit stoichiometry
        elif len(atom) == 3 and not atom[:-1].isalpha():
            symbol = atom[:-2]
            coefficient = int(atom[-2:-1])
        else:
            raise IOError(f" Too many characters in {atom}.")
        return symbol, coefficient

    atom_counts = {}
    for group in _get_groups(molecule):
        for k,v in _get_atoms(group).items():
            if k in atom_counts:
                atom_counts[k] += int(v)
            else:
                atom_counts[k] = v
    return atom_counts


def get_molar_mass(watertap_name: str) -> float:
    """
    Extracts atomic weight data from a periodic table file
    to generate the molar mass of a chemical substance.

    TODO: additional testing for complex solutes
    such as CH3CO2H, [UO2]2[OH]4, etc.

    :param watertap_name: string name of a solute in WaterTAP format

    :return molar_mass: float value for molar mass of solute
    """

    molar_mass = 0
    file_path = Path(__file__).parents[0]
    pt = read_csv(join(file_path, "periodic_table.csv"))
    # split molecule and extract atomic data (exclude charge)
    molecule = watertap_name.split("_")[0]
    atom_counts = get_atom_counts(molecule)
    #print(atom_counts)
    for atom in atom_counts:
        atomic_mass = float(pt["AtomicMass"][(pt["Symbol"] == atom)].values[0])
        molar_mass += atom_counts[atom] * atomic_mass
    if not molar_mass:
        raise IOError(f"Molecular weight data could not be found for {watertap_name}.")
    return molar_mass


def get_molar_mass_quantity(watertap_name: str, units=pyunits.kg / pyunits.mol):
    """
    Extract atomic weight data from a periodic table file
    to generate the molar mass of a chemical substance in pint units.
    Since get_molar_mass returns only the value, which has inherent units of g/mol,
    this function converts to kg/mol by default, the units used for molecular weight by convention in WaterTAP.

    :param watertap_name: string name of a solute in WaterTAP format

    :return desired_quantity: molar mass of solute in pint units. Conversion from g/mol to kg/mol by default.
    """
    molar_mass_value = get_molar_mass(watertap_name)
    inherent_quantity = molar_mass_value * pyunits.g / pyunits.mol
    desired_quantity = pyunits.convert(inherent_quantity, to_units=units)
    return desired_quantity


def get_oli_names(source: dict):
    """
    Update source dictionary with data to populate MCAS property model.

    :param source: dictionary containing WaterTAP names as keys

    :return source: dictionary with OLIName named tuples as keys
    """

    source = dict(
        map(lambda k, v: (watertap_to_oli(k), v), source.keys(), source.values())
    )
    return source


"""
`names_db` is a dictionary of OLI names and their WaterTAP counterparts.

It functions to aid reverse lookup, i.e., if a name is already in OLI format,
the necessary data can still be extracted.

TODO: improve reverse lookup (lookup by group - alkali, halide, etc.)
"""

names_db = {

    "CLION": "Cl_-",

    "H2SO4": "H2[SO4]",
    "HSO4ION": "H[SO4]_-",
    "SO4ION": "[SO4]_2-",
    "SO3": "[SO3]",

    "MGOH2": "Mg[OH]2",
    "MGOHION": "Mg[OH]_-",
    "MGO": "MgO",
    "MGION": "Mg_2+",
    "MGCL2": "MgCl2",
    "MGCO3": "Mg[CO3]",
    "MGSO4": "Mg[SO4]",

    "CAOH2": "Ca[OH]2",
    "CAOHION": "Ca[OH]_-",
    "CAO": "CaO",
    "CAION": "Ca_2+",

    "CAHSO42": "Ca[HSO4]2",
    "CASO4": "Ca[SO4]",
    "CACL2": "CaCl2",
    "CA3CL6.1H2O": "[CaCl2]3.[H2O]",

    "KION": "K_+",
    "KCL": "KCl",
    "K2CO3": "K2[CO3]",
    "KHCO3": "KH[CO3]",
    "K2SO4": "K2[SO4]",
    "KMGSO4ION": "KMg[SO4]_+",

    "NAION": "Na_+",
    "NACL": "NaCl",
    "NAOHCO3ION": "Na[OH][CO3]_2-",
    "NAMGCO32": "NaMg[CO3]2",
    "NAMGSO4ION": "NaMg[SO4]_+",
    "NA2SO4": "Na2[SO4]",
    "NAHSO4": "NaH[SO4]",
    "NAH3SO42": "NaH3[SO4]2",
    "NAH3SO43": "NaH5[SO4]3",
    "NA3HSO42": "Na3H[SO4]2",
    "NA3OHSO4": "Na3[OH][SO4]",

    "HCO3ION": "H[CO3]_-",
    "NA2CO3": "Na2[CO3]",
    "NAHCO3": "NaH[CO3]",
    "CO3ION": "[CO3]_2-",
    "CACO3": "Ca[CO3]",
    "ARAGONITE": "Ca[CO3]",

    "SIO2": "SiO2",
    "H2SIO4ION": "H2SiO4_2-",
    "HSIO3ION": "HSiO3_-",
    "NA2SIO3": "Na2SiO3",
    "AFWILLAM": "Ca3Si2O6[OH]2.[H2O]2",
    "TOBERMAM": "Ca5Si6O17.5.[H2O]5", # need workaround for decimal hydrates

    "CO2": "[CO2]",
    "H2O": "[H2O]",
    "H3OION": "H[H2O]_+", # not sure how best to represent this in WaterTAP
    "HCL": "HCl",

    "KOH": "K[OH]",
    "NAOH": "Na[OH]",
    "OHION": "[OH]_-",


    "PANTIGORIT": "Mg3Si2O5[OH]4",
    "ANTIGORAM": "Mg3Si2O5[OH]4",
    "SEPIOLAM": "Mg4Si6O15[OH]2.[H2O]6",
}


def oli_reverse_lookup(oli_name: str, names_db=names_db) -> OLIName:
    """
    Look up WaterTAP formatted name for solute in OLI format, if listed in names_db dictionary.

    :param oli_name: string name of a solute in OLI format

    :return watertap_name: string name of a solute in WaterTAP format
    """

    if oli_name in names_db:
        watertap_name = names_db[oli_name]
        #print(f"Success (full tag): {oli_name}: {watertap_name}")
        return watertap_name
    else:
        # maybe use a regex to look for .#H2O
        # look for hydrated variants
        oli_name_components = oli_name.split(".")
        base_tag = oli_name_components[0]
        hydrate = oli_name_components[-1]
        num_waters = hydrate.split("H2O")[0]
        if base_tag in names_db:
            watertap_name = names_db[base_tag] + f".[H2O]{num_waters}"
            #print(f"Success (base tag): {oli_name}: {watertap_name}")
            return watertap_name
    #print(f"Warning (no tag): add {oli_name} and watertap_name to names_db.")
    return None
