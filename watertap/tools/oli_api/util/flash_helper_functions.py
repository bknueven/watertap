#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to `eipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
#################################################################################

__author__ = "Paul Vecchiarelli, Ben Knueven"

import json
from itertools import product
from collections import UserDict
from pyomo.environ import units as pyunits, value
import numpy as np

from watertap.tools.oli_api.util.chemistry_helper_functions import get_oli_name, oli_reverse_lookup, get_molar_mass_quantity, get_stoichiometric_expression

import logging

_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "OLIAPI - %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
)
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.INFO)

class FixedKeysDict(UserDict):
    def __init__(self, d):
        self.data = d

    def __setitem__(self, k, v):
        if k not in self.data:
            raise RuntimeError(f" Key {k} not in dictionary.")
        else:
            self.data[k] = v

    def __delitem__(self, k):
        raise Exception(" Deleting keys not supported for this object.")

    def _check_value(self, k, valid_values):
        if k not in self.data:
            raise RuntimeError(f" Key {k} not in dictionary.")
        else:
            if self.data[k] not in valid_values:
                raise RuntimeError(
                    f" Value {self.data[k]} not a valid value for key {k}."
                )

    def pprint(self):
        print("-------------------------")
        for key, value in self.data.items():
            print(f" {key}\n - {value}\n")


input_unit_set = FixedKeysDict(
    {
        "mass_concentration": {
            "oli_unit": "mg/L",
            "pyomo_unit": pyunits.mg / pyunits.L,
        },
        "molecularConcentration": {
            "oli_unit": "mg",
            "pyomo_unit": pyunits.mg,
        },
        "mass": {"oli_unit": "mg", "pyomo_unit": pyunits.mg},
        "temperature": {"oli_unit": "K", "pyomo_unit": pyunits.K},
        "pressure": {"oli_unit": "Pa", "pyomo_unit": pyunits.Pa},
        "enthalpy": {"oli_unit": "J", "pyomo_unit": pyunits.J},
        "vaporAmountMoles": {"oli_unit": "mol", "pyomo_unit": pyunits.mol},
        "vaporMolFrac": {
            "oli_unit": "mol/mol",
            "pyomo_unit": pyunits.mol / pyunits.mol,
        },
        "totalVolume": {"oli_unit": "L", "pyomo_unit": pyunits.L},
        "pipeDiameter": {"oli_unit": "m", "pyomo_unit": pyunits.meter},
        "pipeFlowVelocity": {
            "oli_unit": "m/s",
            "pyomo_unit": pyunits.meter / pyunits.second,
        },
        "diskDiameter": {"oli_unit": "m", "pyomo_unit": pyunits.meter},
        "diskRotatingSpeed": {"oli_unit": "cycle/s", "pyomo_unit": 1 / pyunits.second},
        "rotorDiameter": {"oli_unit": "m", "pyomo_unit": pyunits.meter},
        "rotorRotation": {"oli_unit": "cycle/s", "pyomo_unit": 1 / pyunits.second},
        "shearStress": {"oli_unit": "Pa", "pyomo_unit": pyunits.Pa},
        "pipeDiameter": {"oli_unit": "m", "pyomo_unit": pyunits.meter},
        "pipeRoughness": {"oli_unit": "m", "pyomo_unit": pyunits.meter},
        "liquidFlowInPipe": {
            "oli_unit": "L/s",
            "pyomo_unit": pyunits.L / pyunits.second,
        },
        "gasFlowInPipe": {"oli_unit": "L/s", "pyomo_unit": pyunits.L / pyunits.second},
        "viscAbs2ndLiq": {
            "oli_unit": "Pa-s",
            "pyomo_unit": pyunits.Pa * pyunits.second,
        },
        "alkalinity": {"oli_unit": "mg HCO3/L", "pyomo_unit": pyunits.mg / pyunits.L},
        "TIC": {"oli_unit": "mol C/L", "pyomo_unit": pyunits.mol / pyunits.L},
        "CO2GasFraction": {
            "oli_unit": "mol/mol",
            "pyomo_unit": pyunits.mol / pyunits.mol,
        },
    }
)

optional_properties = FixedKeysDict(
    {
        "electricalConductivity": True,
        "viscosity": True,
        "selfDiffusivityAndMobility": True,
        "heatCapacity": True,
        "thermalConductivity": True,
        "surfaceTension": True,
        "interfacialTension": True,
        "volumeStdConditions": True,
        "prescalingTendenciesEstimated": False,
        "prescalingIndexEstimated": False,
        "prescalingTendenciesRigorous": True,
        "prescalingIndexRigorous": True,
        "scalingTendencies": True,
        "scalingIndex": True,
        "hardness": True,
        "ionicStrengthXBased": True,
        "ionicStrengthMBased": True,
        "totalDissolvedSolids": True,
        "vaporToInflowMoleFraction": True,
        "partialPressure": True,
        "vaporDiffusivityMatrix": True,
        "entropyStream": True,
        "entropySpecies": True,
        "entropyStreamStandardState": True,
        "entropySpeciesStandardState": True,
        "gibbsEnergyStream": True,
        "gibbsEnergySpecies": True,
        "gibbsEnergyStreamStandardState": True,
        "gibbsEnergySpeciesStandardState": True,
        "activityCoefficientsXBased": True,
        "activityCoefficientsMBased": True,
        "fugacityCoefficients": True,
        "vaporFugacity": True,
        "kValuesXBased": True,
        "kValuesMBased": True,
        "MBGComposition": True,
        "materialBalanceGroup": True,
    }
)

# TODO: consider adding https://devdocs.olisystems.com/user-defined-output-unit-set
output_unit_set = FixedKeysDict(
    {
        "enthalpy": input_unit_set["enthalpy"]["oli_unit"],
        "mass": input_unit_set["mass"]["oli_unit"],
        "pt": input_unit_set["pressure"]["oli_unit"],
        "total": input_unit_set["mass"]["oli_unit"],
        "liq1_phs_comp": input_unit_set["mass"]["oli_unit"],
        "solid_phs_comp": input_unit_set["mass"]["oli_unit"],
        "vapor_phs_comp": input_unit_set["mass"]["oli_unit"],
        "liq2_phs_comp": input_unit_set["mass"]["oli_unit"],
        "combined_phs_comp": input_unit_set["mass"]["oli_unit"],
        "molecularConcentration": input_unit_set["molecularConcentration"]["oli_unit"],
    }
)


def write_output(content, file_name):
    """
    Write dictionary-based content to file.

    :param content: dictionary of content to write
    :param file_name: string for name of file to write
    """

    _logger.info(f"Saving content to {file_name}")
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(content, f)
    _logger.info("Save complete")


def build_survey(survey_arrays, get_oli_names=False, file_name=None, mesh_grid=True):
    """
    Build a dictionary for modifying flash calculation parameters.

    :param survey_arrays: dictionary for variables and values to survey
    :param get_oli_names: bool switch to convert name into OLI name
    :param file_name: string for file to write, if any
    :param mesh_grid: if True (default) the input array will be combined to generate combination of all possible samples
        if False, the direct values in survey_arrays will be used

    :return survey: dictionary for product of survey variables and values
    """
    _name = lambda k: get_oli_name(k) if get_oli_names else k
    if mesh_grid:
        keys = [get_oli_name(k) if get_oli_names else k for k in survey_arrays]
        values = list(product(*(survey_arrays.values())))
        survey = {_name(keys[i]): [val[i] for val in values] for i in range(len(keys))}
    else:
        survey = {}
        values = None
        for key, arr in survey_arrays.items():
            survey[_name(key)] = arr
            if values is not None and len(values) != len(arr):
                raise ValueError(f"Length of list for key {key} differs from prior key")
            values = arr
    _logger.info(f"Survey contains {len(values)} items.")
    if file_name:
        write_output(survey, file_name)
    return survey


def get_survey_sample_conditions(survey, sample_points):
    """
    Return survey parameter values for one or more sample points.

    :param survey: dictionary for product of survey conditions and values
    :param sample_points: list of indices to get parameter values from

    :return sample_conditions: dictionary for parameter values for given samples
    """

    sample_conditions = {}
    for point in sample_points:
        sample_conditions[point] = {}
        for k, v in survey.items():
            sample_conditions[point][k] = v[point]
    _logger.debug(sample_conditions)
    return sample_conditions


def get_solids_formed(survey_result):
    """
    Get the mass of solids formed for samples in an OLIApi survey.

    :param survey_results: JSON (OLIApi stream output) containing a solid phase.

    :return solids_formed: dictionary mapping precipitates and the amount of solid formed for each survey sample (in kilograms/L)
    """

    solids_formed = {}
    volume_location = survey_result["result"]["volume_total"]
    volume_unit = input_unit_set["totalVolume"]
    print(f"Total volume in units of {volume_unit['oli_unit']}")
    solids_mass_per_liter = survey_result["result"]["molecularConcentration_solid"]
    total_volume = survey_result["result"]["volume_total"]["values"]
    for solid in solids_mass_per_liter:
        concentrations = []
        mass_per_liter = solids_mass_per_liter[solid]["values"]
        for idx, v in enumerate(mass_per_liter):
            mass_quantity = pyunits.convert(
                mass_per_liter[idx]*input_unit_set["molecularConcentration"]["pyomo_unit"],
                to_units=pyunits.kg,
            )
            volume_quantity = volume_location["values"][idx]*volume_unit["pyomo_unit"]
            concentration = mass_quantity/volume_quantity
            concentrations.append(concentration)
        if any(value(concentration) != 0 for concentration in concentrations):
            solids_formed[oli_reverse_lookup(solid)] = concentrations
    return solids_formed


"""
`reactions` provides a dictionary of minerals and the components they dissolve into. This method may be used with `get_stoichiometric_expression` to determine the stoichiometry of some simpler reactions.

TODO: improve dissolution and precipitation modeling for chemistry functions
"""
reactions = {
    "Ca[CO3]": ["Ca_2+", "[CO3]_2-"],
    "Mg[OH]2": ["Mg_2+", "[OH]_-"],
    "Ca[OH]2": ["Ca_2+", "[OH]_-"],
}


def calculate_solid_phase_loss(solids_formed, reactions=reactions):
    """
    Calculate the change in mass of solutes given a set of precipitates and related reactions.

    :param solids_formed: dictionary mapping precipitates and the amount of solid formed for each survey sample
    :param reactions: dictionary mapping precipitates with solutes formed from dissolution

    :return solid_phase_loss: dictionary mapping solutes and the change in mass (assumed per liter) due to precipitation
    """

    solid_phase_loss = {}
    reactions_present = {
        solid:reactions[solid] for solid in solids_formed if solid in reactions
    }
    for solid, solutes in reactions_present.items():
        molar_mass_qtys = {solid: get_molar_mass_quantity(solid)}
        for solute in solutes:
            molar_mass_qtys[solute] = get_molar_mass_quantity(solute)
        reaction_stoichiometry = get_stoichiometric_expression(solid, solutes)
        reaction_stoichiometry.pop(solid)
        # create numpy arrays
        array = np.array(solids_formed[solid])
        moles_lost_solid = array / molar_mass_qtys[solid]
        for solute, stoich in reaction_stoichiometry.items():
            moles_lost_solute = moles_lost_solid * stoich
            mass_lost_solute = moles_lost_solute * molar_mass_qtys[solute]
            if solute not in solid_phase_loss:
                solid_phase_loss[solute] = mass_lost_solute
            else:
                solid_phase_loss[solute] += mass_lost_solute
    return solid_phase_loss
