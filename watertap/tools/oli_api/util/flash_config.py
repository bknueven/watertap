import logging

from watertap.tools.oli_api.util.flash_helper_functions import (
    input_unit_set,
    output_unit_set,
    optional_properties,
    write_output
)
from watertap.tools.oli_api.util.chemistry_helper_functions import (
    watertap_to_oli,
    get_oli_name,
)

_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "OLIAPI - %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
)
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.INFO)

def configure_json_input(flash_method, **kwargs):
    """
    Configure flash analysis JSON from keyword inputs.

    :param flash_method: string for flash calculation name
    """

    if flash_method == "wateranalysis":
        json_input = configure_water_analysis(**kwargs)
    else:
        json_input = configure_flash_analysis(**kwargs)
    return json_input

def configure_water_analysis(
    inflows=None,
    temperature=None,
    pressure=None,
    reconciliation=None,
    electroneutrality=None,
    makeup_ion=None,
    ph=None,
    acid_titrant=None,
    base_titrant=None,
    alkalinity=None,
    alkalinity_ph=None,
    alkalinity_titrant=None,
    tic=None,
    allow_solids=False,
    included_solids=None,
    excluded_solids=None,
    calc_alkalinity=False,
    use_scaling_rigorous=True,
    file_name=None,
):
    """
    Configure Water Analysis JSON input.

    :param inflows: dictionary of solutes
    :param temperature: float for temperature in Kelvins
    :param pressure: float for pressure in Pascals
    :param reconciliation: string for method of reconciliation: "EquilCalcOnly" (default), "ReconcilePh", "ReconcilePhAndAlkalinity", or "ReconcilePhAndAlkalinityAndTic"; "ReconcileCo2Gas" not supported currently.
    :param electroneutrality: string for method of electroneutrality calculation: "DominantIon", "ProrateCations", "ProrateAnions", "Prorate", "AutoNACL", or "MakeupIon" are supported
    :param makeup_ion: string for ion to use for electroneutrality balance, if "MakeupIon,
    :param ph: float for pH to reconcile solution to, required for pH based reconciliation
    :param acid_titrant: string for acidification titrant, used in pH based reconciliation
    :param base_titrant: string for basification titrant, used in pH based reconciliation
    :param alkalinity: float for alkalinity to reconcile solution to, required for Alk based reconciliation
    :param alkalinity_ph: float for alkalinity endpoint ph, used in Alk based reconciliation
    :param alkalinity_titrant: string for alkalinity titration species, used in Alk based reconciliation
    :param tic: float for total inorganic carbon concentration to reconcile solution to, required for TIC based reconcilation
    :param allow_solids: bool to enable solid phase formation
    :param included_solids: list of solids to include in analysis
    :param excluded_solids: list of solids to exclude from analysis
    :param calc_alkalinity: bool to calculate alkalinity of solution
    :param use_scaling_rigorous: bool to switch between Rigorous (default) and Estimated scaling computations
    :param file_name: string for file to write, if any
    :param mesh_grid: if True (default) the input array will be combined to generate combination of all possible samples
        if False, the direct values in survey_arrays will be used

    :return json_input: JSON for Water Analysis
    """

    _logger.info("Configuring Water Analysis JSON ...")
    input_list = []

    if not inflows:
        raise RuntimeError("Inflows must be defined for Water Analysis.")

    temp_input = {
        "group": "Properties",
        "name": "Temperature",
        "unit": input_unit_set["temperature"]["oli_unit"],
        "value": 273.15,
    }
    if temperature is not None:
        if float(temperature):
            temp_input.update({"value": float(temperature)})
        else:
            raise ValueError(f"Invalid temperature: {temperature}. Expected number")
    input_list.append(temp_input)

    pres_input = {
        "group": "Properties",
        "name": "Pressure",
        "unit": input_unit_set["pressure"]["oli_unit"],
        "value": 101325,
    }
    if pressure is not None:
        if float(pressure):
            pres_input.update({"value": float(pressure)})
        else:
            raise ValueError(f"Invalid pressure: {pressure}. Expected number")
    input_list.append(pres_input)

    reconciliation_options = [
        "EquilCalcOnly",
        "ReconcilePh",
        "ReconcilePhAndAlkalinity",
        "ReconcilePhAndAlkalinityAndTic",
    ]
    rec_input = {
        "group": "Calculation Options",
        "name": "CalcType",
        "value": "EquilCalcOnly",
    }
    if reconciliation is not None:
        if reconciliation in reconciliation_options:
            rec_input.update({"value": reconciliation})
        else:
            raise RuntimeError(
                f"Invalid reconciliation option: {reconciliation}."
                + f" Use one of {reconciliation_options}"
            )
    else:
        reconciliation = "EquilCalcOnly"

    input_list.append(rec_input)
    additional_req_input = []
    additional_req_args = []
    if "Ph" in reconciliation:
        additional_req_args.append([ph, acid_titrant, base_titrant])
        if not acid_titrant:
            acid_titrant = "HCl"
        if not base_titrant:
            base_titrant = "NaOH"
        additional_req_input.extend(
            [
                {
                    "group": "Properties",
                    "name": "pH",
                    "value": ph,
                },
                {
                    "group": "Calculation Options",
                    "name": "PhAcidTitrant",
                    "value": get_oli_name(acid_titrant),
                },
                {
                    "group": "Calculation Options",
                    "name": "PhBaseTitrant",
                    "value": get_oli_name(base_titrant),
                },
            ]
        )
    if "Alk" in reconciliation:
        additional_req_args.append([alkalinity, alkalinity_ph, alkalinity_titrant])
        if not alkalinity_titrant:
            alkalinity_titrant = "H2SO4"
        if not alkalinity_ph:
            alkalinity_ph = 4.5
            _logger.info("No alkalinity endpoint pH specified. Assuming 4.5.")
            additional_req_input.extend(
                [
                    {
                        "group": "Properties",
                        "name": "Alkalinity",
                        "unit": input_unit_set["alkalinity"]["oli_unit"],
                        "value": alkalinity,
                    },
                    {
                        "group": "Properties",
                        "name": "AlkalinityTitrationEndPointpH",
                        "value": alkalinity_ph,
                    },
                    {
                        "group": "Calculation Options",
                        "name": "AlkalinityPhTitrant",
                        "value": alkalinity_titrant,
                    },
                ]
            )
    if "Tic" in reconciliation:
        additional_req_args.append([tic])
        additional_req_input.append(
            {
                "group": "Properties",
                "name": "TIC",
                "unit": input_unit_set["TIC"]["oli_unit"],
                "value": tic,
            }
        )
    missing_keys = [arg for arg in additional_req_args if arg is None]
    if missing_keys:
        raise RuntimeError(f"Missing keys for {reconciliation}: {missing_keys}")
    input_list.extend(additional_req_input)

    electroneutrality_options = [
        "DominantIon",
        "ProrateCations",
        "ProrateAnions",
        "Prorate",
        "AutoNACL",
        "MakeupIon",
    ]
    elec_input = {
        "group": "Electroneutrality Options",
        "name": "ElectroNeutralityBalanceType",
        "value": "DominantIon",
    }

    if electroneutrality is not None:
        if electroneutrality in electroneutrality_options:
            elec_input.update({"value": electroneutrality})
        else:
            raise RuntimeError(
                f"Invalid reconciliation option: {electroneutrality}."
                + f" Use one of {electroneutrality_options}"
            )
    input_list.append(elec_input)
    if electroneutrality == "MakeupIon":
        if makeup_ion is not None:
            input_list.append(
                {
                    "group": "Electroneutrality Options",
                    "name": "MakeupIonBaseTag",
                    "value": get_oli_name(makeup_ion),
                }
            )

    input_list.extend(
        [
            {
                "group": "Calculation Options",
                "name": "AllowSolidsToForm",
                "value": bool(allow_solids),
            },
            {
                "group": "Calculation Options",
                "name": "CalcAlkalnity",
                "value": bool(calc_alkalinity),
            },
        ]
    )
    conc_unit = input_unit_set["mass_concentration"]["oli_unit"]
    _logger.info(f"Using {conc_unit} for inflows input")
    for k, v in inflows.items():
        k_oli = watertap_to_oli(k)
        if k not in ["H2O", "H3O_+", "H_+"]:
            input_list.append(
                {
                    "group": k_oli.charge_group,
                    "name": k_oli.oli_name,
                    "unit": conc_unit,
                    "value": v,
                    "charge": k_oli.charge,
                }
            )

    json_input = _add_to_json(
        "wateranalysis",
        input_list,
        included_solids,
        excluded_solids,
        use_scaling_rigorous,
        file_name,
    )
    return json_input

def configure_flash_analysis(
    inflows=None,
    flash_method=None,
    temperature=None,
    pressure=None,
    calculated_variable=None,
    enthalpy=None,
    vapor_amount=None,
    vapor_fraction=None,
    volume=None,
    ph=None,
    acid_titrant=None,
    base_titrant=None,
    formed_solid=None,
    precipitant_inflow=None,
    included_solids=None,
    excluded_solids=None,
    contact_surface=None,
    flow_type=None,
    diameter=None,
    liq_velocity=None,
    gas_velocity=None,
    rot_velocity=None,
    shear_stress=None,
    roughness=None,
    nonaqueous_visc=None,
    water_cut_inversion=None,
    relative_visc_inversion=None,
    use_scaling_rigorous=True,
    file_name=None,
):

    """
    Configure Flash Analysis JSON input.

    :param inflows: dictionary of solutes, of the form {"unit": unit, "values": {solute: concentration}}
    :param flash_method: string for flash calculation name
    :param temperature: float for temperature in Kelvins
    :param pressure: float for pressure in Pascals
    :param calculated_variable: string for variable to calculate, such as temperature or pressure, used in 'bubblepoint', 'dewpoint', 'vapor-amount', 'vapor-fraction', and 'isochoric' flashes
    :param enthalpy: float for total enthalpy in Joules, used in 'isenthalpic' flash
    :param vapor_amount: float for vapor phase Moles, used in 'vapor-amount' flash
    :param vapor_fraction: float for vapor phase in Mole %, used in 'vapor-fraction' flash
    :param volume: float for total volume in Cubic Meters, used in 'isochoric' flash
    :param ph: float for target pH, used in 'setph' flash
    :param acid_titrant: string for acidification titrant, used in 'setph' flash
    :param base_titrant: string for basification titrant, used in 'setph' flash
    :param formed_solid: string for solid species to precipitate based on inflow sweep, used in 'precipitation-point'
    :param precipitant_inflow: string for inflow species to sweep, used in 'precipitation-point'
    :param included_solids: list of solids to include in analysis
    :param excluded_solids: list of solids to exclude from analysis
    :param contact_surface: string for contact surface metal name
    :param flow_type: string for flow configuration
    :param diameter: float for diameter of surface (i.e., pipe or rotor)
    :param liq_velocity: float for velocity of liquid flow
    :param gas_velocity: float for velocity of vapor flow, used in 'approximateMultiPhaseFlow'
    :param rot_velocity: float for rotational velocity
    :param shear_stress: float for defined shear stress, used in 'definedShearStress'
    :param roughness: float for pipe roughness, used in 'approximateMultiPhaseFlow'
    :param nonaqueous_visc: float for absolute viscosity of nonaqueous phase, used in 'approximateMultiPhaseFlow'
    :param water_cut_inversion: float for water cut at point of dispersion inversion, used in 'approximateMultiPhaseFlow'
    :param relative_visc_inversion: float for maximum relative viscosity of dispersion at inversion, used in 'approximateMultiPhaseFlow'
    :param use_scaling_rigorous: bool to switch between Rigorous (default) and Estimated scaling computations
    :param file_name: string for file to write, if any

    :return json_input: JSON for Water Analysis
    """

    _logger.info(f"Configuring {flash_method} Flash JSON ...")

    if flash_method not in [
        "isothermal",
        "isenthalpic",
        "bubblepoint",
        "dewpoint",
        "vapor-amount",
        "vapor-fraction",
        "isochoric",
        "setph",
        "precipitation-point",
        "corrosion-rates",
    ]:
        raise RuntimeError(
            f"Failed to configure Flash. Invalid method: {flash_method}"
        )

    if not inflows:
        raise RuntimeError("Inflows must be defined for Flash Analysis.")

    input_dict = {}
    temp_input = {
        "unit": input_unit_set["temperature"]["oli_unit"],
        "value": 273.15,
    }
    if temperature:
        if float(temperature):
            temp_input.update({"value": float(temperature)})
        else:
            raise ValueError(f"Invalid temperature: {temperature}. Expected number")
    input_dict["temperature"] = temp_input

    pres_input = {
        "unit": input_unit_set["pressure"]["oli_unit"],
        "value": 101325,
    }
    if pressure:
        if float(pressure):
            pres_input.update({"value": float(pressure)})
        else:
            raise ValueError(f"Invalid pressure: {pressure}. Expected number")
    input_dict["pressure"] = pres_input

    if flash_method in [
        "bubblepoint",
        "dewpoint",
        "vapor-amount",
        "vapor-fraction",
        "isochoric",
    ]:
        if calculated_variable is not None:
            if calculated_variable not in ["temperature", "pressure"]:
                raise RuntimeError(
                    f"Invalid input for 'calculated_variable': {calculated_variable}; 'temperature' or 'pressure' supported."
                )
            _logger.info(
                f"{flash_method} will calculate {calculated_variable} as its variable"
            )
            input_dict["calculatedVariable"] = calculated_variable
        else:
            raise RuntimeError(
                f"Missing argument for {flash_method}: 'calculated_variable'"
            )

    if flash_method == "isenthalpic":
        enth_input = {
            "unit": input_unit_set["enthalpy"]["oli_unit"],
            "value": None,
        }
        if float(enthalpy):
            enth_input.update({"value": float(enthalpy)})
        else:
            raise ValueError(f"Invalid enthalpy: {enthalpy}. Expected number")
        input_dict["enthalpy"] = enth_input

    if flash_method == "vapor-amount":
        vapor_amount_input = (
            {
                "unit": input_unit_set["vaporAmountMoles"]["oli_unit"],
                "value": None,
            },
        )
        if float(vapor_amount):
            vapor_amount_input.update({"value": float(vapor_amount)})
        else:
            raise ValueError(
                f"Invalid vapor amount: {vapor_amount}. Expected number"
            )
        input_dict["vaporAmountMoles"] = vapor_amount_input

    if flash_method == "vapor-fraction":
        vapor_fraction_amount = (
            {
                "unit": input_unit_set["vaporMolFrac"]["oli_unit"],
                "value": None,
            },
        )
        if float(vapor_fraction):
            vapor_fraction_amount.update({"value": float(vapor_fraction)})
        else:
            raise ValueError(
                f"Invalid vapor fraction: {vapor_fraction}. Expected number"
            )
        input_dict["vaporMolFrac"] = vapor_fraction_input

    if flash_method == "isochoric":
        volume_input = {
            "unit": input_unit_set["totalVolume"]["oli_unit"],
            "value": None,
        }
        if float(volume):
            volume_input.update({"value": float(volume)})
        else:
            raise ValueError(f"Invalid volume: {volume}. Expected number")
        input_dict["totalVolume"] = volume_input

    if flash_method == "setph":
        ph_input = {
            "targetPH": {
                "unit": "",
                "value": None,
            },
        }
        if float(ph):
            ph_input["targetPH"].update({"value": float(ph)})
        else:
            raise ValueError(f"Invalid ph: {ph}. Expected number")
        input_dict["targetPH"] = ph_input
        if not acid_titrant:
            acid_titrant = "HCl"
        input_dict["pHAcidTitrant"] = get_oli_name(acid_titrant)
        if not base_titrant:
            base_titrant = "NaOH"
        input_dict["pHBaseTitrant"] = get_oli_name(base_titrant)

    if flash_method == "precipitation-point":
        missing_args = [
            arg for arg in [formed_solid, precipitant_inflow] if arg is None
        ]
        if missing_args:
            raise RuntimeError(
                f"Missing argument(s) for {flash_method}: {missing_args}"
            )
        else:
            input_dict.update(
                {
                    "solidToPrecipitate": formed_solid,
                    "inflowToAdjust": precipitant_inflow,
                }
            )

    input_dict["inflows"] = inflows

    if flash_method == "corrosion-rates":
        _logger.info(
            f"Ensure DBS file uses 'AQ' thermodynamic framework to use Corrosion Analyzer"
        )
        input_dict["corrosionParameters"] = _configure_corrosion(
            contact_surface,
            flow_type,
            diameter,
            liq_velocity,
            gas_velocity,
            rot_velocity,
            shear_stress,
            roughness,
            nonaqueous_visc,
            water_cut_inversion,
            relative_visc_inversion,
        )

    json_input = _add_to_json(
        flash_method,
        input_dict,
        included_solids,
        excluded_solids,
        use_scaling_rigorous,
        file_name,
    )
    return json_input

def _configure_corrosion(
    contact_surface,
    flow_type,
    diameter,
    liq_velocity,
    gas_velocity,
    rot_velocity,
    shear_stress,
    roughness,
    nonaqueous_visc,
    water_cut_inversion,
    relative_visc_inversion,
):
    """
    Configure input dict for Corrosion Rates flash.

    :param contact_surface: string for contact surface metal name
    :param flow_type: string for flow configuration
    :param diameter: float for diameter of surface (i.e., pipe or rotor)
    :param liq_velocity: float for velocity of liquid flow
    :param gas_velocity: float for velocity of vapor flow, used in 'approximateMultiPhaseFlow'
    :param rot_velocity: float for rotational velocity
    :param shear_stress: float for defined shear stress, used in 'definedShearStress'
    :param roughness: float for pipe roughness, used in 'approximateMultiPhaseFlow'
    :param nonaqueous_visc: float for absolute viscosity of nonaqueous phase, used in 'approximateMultiPhaseFlow'
    :param water_cut_inversion: float for water cut at point of dispersion inversion, used in 'approximateMultiPhaseFlow'
    :param relative_visc_inversion: float for maximum relative viscosity of dispersion at inversion, used in 'approximateMultiPhaseFlow'

    :return config: dictionary for corrosion analysis parameters
    """

    valid_flow_types = [
        "static",
        "pipeFlow",
        "rotatingDisk",
        "rotatingCylinder",
        "completeAgitation",
        "definedShearStress",
        "approximateMultiPhaseFlow",
    ]
    if flow_type not in valid_flow_types:
        raise RuntimeError(
            f"Invalid flow_type: {flow_type}."
            f"Expected one of {', '.join(t for t in valid_flow_types)}"
        )

    config = {
        "calculationType": "isothermal",
        "corrosionParameters": {
            "contactSurface": contact_surface,
            "flowType": flow_type,
        },
    }

    def _try_float(v):
        try:
            val = float(v)
        except:
            val = None
        return val

    _check_args = lambda args: [arg for arg in args if arg is None]
    _check_floats = lambda args: [arg for arg in args if _try_float(arg) is None]
    if flow_type == "pipeFlow":
        args = [diameter, liq_velocity]
        missing_args = _check_args(args)
        not_floats = _check_floats(args)
    elif flow_type in ["rotatingDisk", "rotatingCylinder"]:
        args = [diameter, rot_velocity]
        missing_args = _check_args(args)
        not_floats = _check_floats(args)
    elif flow_type == "definedShearStress":
        args = [shear_stress]
        missing_args = _check_args(args)
        not_floats = _check_floats(args)
    elif flow_type == "approximateMultiPhaseFlow":
        args = [
            diameter,
            liq_velocity,
            gas_velocity,
            roughness,
            nonaqueous_visc,
            water_cut_inversion,
            relative_visc_inversion,
        ]
        missing_args = _check_args(args)
        not_floats = _check_floats(args)
    if missing_args:
        raise RuntimeError(
            f"Missing argument(s) for {flash_method}: {missing_args}"
        )
    if not_floats:
        raise RuntimeError(
            f"Invalid values for argument(s): {not_floats}. Expected value"
        )

    if flow_type == "pipeFlow":
        config["corrosionParameters"].update(
            {
                "pipeDiameter": {
                    "value": diameter,
                    "unit": input_unit_set["pipeDiameter"]["oli_unit"],
                },
                "pipeFlowVelocity": {
                    "value": liq_velocity,
                    "unit": input_unit_set["pipeFlowVelocity"]["oli_unit"],
                },
            }
        )
    elif flow_type == "rotatingDisk":
        config["corrosionParameters"].update(
            {
                "diskDiameter": {
                    "value": diameter,
                    "unit": input_unit_set["diskDiameter"]["oli_unit"],
                },
                "diskRotationSpeed": {
                    "value": rot_velocity,
                    "unit": input_unit_set["diskRotationSpeed"]["oli_unit"],
                },
            },
        )

    elif flow_type == "rotatingCylinder":
        config["corrosionParameters"].update(
            {
                "rotorDiameter": {
                    "value": diameter,
                    "unit": input_unit_set["rotorDiameter"]["oli_unit"],
                },
                "rotorRotation": {
                    "value": rot_velocity,
                    "unit": input_unit_set["rotorRotation"]["oli_unit"],
                },
            },
        )

    elif flow_type == "definedShearStress":
        config["corrosionParameters"].update(
            {
                "shearStress": {
                    "value": shear_stress,
                    "unit": input_unit_set["shearStress"]["oli_unit"],
                },
            },
        )

    elif flow_type == "approximateMultiPhaseFlow":
        config["corrosionParameters"].update(
            {
                "pipeDiameter": {
                    "value": diameter,
                    "unit": input_unit_set["pipeDiameter"]["oli_unit"],
                },
                "liquidFlowInPipe": {
                    "value": liq_velocity,
                    "unit": input_unit_set["liquidFlowInPipe"]["oli_unit"],
                },
                "gasFlowInPipe": {
                    "value": gas_velocity,
                    "unit": input_unit_set["gasFlowInPipe"]["oli_unit"],
                },
                "pipeRoughness": {
                    "value": roughness,
                    "unit": input_unit_set["pipeRoughness"]["oli_unit"],
                },
                "viscAbs2ndLiq": {
                    "value": nonaqueous_visc,
                    "unit": input_unit_set["viscAbs2ndLiq"]["oli_unit"],
                },
                "waterCutAtPointOfDispersionInversion": water_cut_inversion,
                "maxRelViscosityOfDispersionAtInversion": relative_visc_inversion,
            }
        )
    return config

def _add_to_json(
    flash_method,
    input_data,
    included_solids,
    excluded_solids,
    use_scaling_rigorous,
    file_name,
):
    """
    Add input data to JSON.

    :param flash_method: string for flash calculation name
    :param input_data: data object from flash configuration function
    :param included_solids: list of solids to include in analysis
    :param excluded_solids: list of solids to exclude from analysis
    :param use_scaling_rigorous: bool to switch between Rigorous (default) and Estimated scaling computations
    :param file_name: string for file to write, if any
    """

    def _set_prescaling_mode(use_scaling_rigorous):
        props = optional_properties
        props
        if bool(use_scaling_rigorous) != bool(props["prescalingTendenciesRigorous"]):
            new_values = {k: not v for k, v in props.items() if "prescaling" in k}
            props.update(new_values)
        return props

    props = _set_prescaling_mode(use_scaling_rigorous)
    json_input = {"params": {}}

    if flash_method == "wateranalysis":
        json_input["params"].update({"waterAnalysisInputs": input_data})
    else:
        json_input["params"] = input_data

    additional_params = {
        "optionalProperties": dict(props),
        "unitSetInfo": dict(output_unit_set),
    }
    if included_solids and excluded_solids:
        raise RuntimeError(
            "Invalid argument combination. "
            "Only one of included_solids and excluded_solids "
            "may be specified at once."
        )
    else:
        if included_solids:
            additional_params.update({"included_solids": list(included_solids)})
        if excluded_solids:
            additional_params.update({"excluded_solids": list(excluded_solids)})
    json_input["params"].update(additional_params)

    if file_name is not None:
        write_output(json_input, file_name)
    return json_input
