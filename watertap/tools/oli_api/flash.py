###############################################################################
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
# OLI Systems, Inc. Copyright © 2022, all rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# 3. Neither the name of OLI Systems, Inc. nor the names of any contributors to
# the software made available herein may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the
# features, functionality or performance of the source code ("Enhancements") to anyone; however,
# if you choose to make your Enhancements available either publicly, or directly to OLI Systems, Inc.,
# without imposing a separate written license agreement for such Enhancements, then you hereby grant
# the following license: a non-exclusive, royalty-free perpetual license to install, use, modify, prepare
# derivative works, incorporate into other computer software, distribute, and sublicense such enhancements
# or derivative works thereof, in binary and source code form.
###############################################################################

__author__ = "Oluwamayowa Amusat, Alexander Dudchenko, Paul Vecchiarelli"


import logging

import json

from copy import deepcopy
from itertools import product

from watertap.tools.oli_api.util.chemistry_helper_functions import get_oli_name

from watertap.tools.oli_api.util.postprocessing import _flatten_results

from watertap.tools.oli_api.util.flash_helper_functions import (
    input_unit_set,
    write_output,
    build_survey,
    get_survey_sample_conditions,
)

from watertap.tools.oli_api.util.flash_config import configure_json_input

from watertap.tools.oli_api.client import OLIApi

_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "OLIAPI - %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
)
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.DEBUG)


class Flash:
    """
    A class to execute OLI Cloud flash calculations.

    :param relative_inflows: bool switch for surveys - true to add specified value to initial value, false to replace initial value with specified value
    :param debug_level: string defining level of logging activity
    """

    def __init__(
        self,
        relative_inflows=True,
        debug_level="INFO",
    ):
        self.relative_inflows = relative_inflows
        if debug_level == "INFO":
            _logger.setLevel(logging.INFO)
        else:
            _logger.setLevel(logging.DEBUG)

        self.build_survey = build_survey
        self.get_survey_sample_conditions = get_survey_sample_conditions
        self.configure_json_input = configure_json_input

    # TODO: something about burst tag seems not to work correctly
    def run_flash(
        self,
        flash_method,
        oliapi_instance,
        dbs_file_id,
        json_input,
        survey=None,
        file_name="",
        #max_concurrent_processes=1000,
        burst_job_tag=None,
        batch_size=None,
    ):
        """
        Conduct single point analysis with initial JSON input, or conduct a survey on that input.

        :param flash_method: string for flash calculation name
        :param oliapi_instance: instance of OLI Cloud API
        :param dbs_file_id: string ID of DBS file
        :param json_input: JSON input for flash calculation
        :param survey: dictionary containing names and input values to modify in JSON
        :param file_name: string for file to write, if any

        :return processed_requests: results from processed OLI flash requests
        """

        if self.relative_inflows:
            _logger.info(
                f"relative_inflows={self.relative_inflows},"
                + " surveys will add values to initial state"
            )

        if flash_method == "corrosion-rates":
            # check if DBS file is using AQ thermodynamic framework
            oliapi_instance.get_corrosion_contact_surfaces(dbs_file_id)

        if survey is None:
            survey = {}
        num_samples = None
        for k, v in survey.items():
            if num_samples is None:
                num_samples = len(v)
            elif num_samples != len(v):
                raise RuntimeError(f"Length of list for key {k} differs from prior key")
        if num_samples is None:
            num_samples = 1
        requests_to_process = []
        for idx in range(num_samples):
            requests_to_process.append(
                {
                    "flash_method": flash_method,
                    "dbs_file_id": dbs_file_id,
                    "input_params": self.get_clone(
                        flash_method, json_input, idx, survey
                    ),
                }
            )
        _logger.info(f"Processing {len(requests_to_process)} OLIApi requests ...")
        processed_requests = oliapi_instance.process_request_list(
            requests_to_process,
            burst_job_tag=burst_job_tag,
            #max_concurrent_processes=max_concurrent_processes,
            batch_size=batch_size,
        )
        _logger.info("Completed running flash calculations")
        result = _flatten_results(processed_requests)
        if file_name:
            write_output(result, file_name)
        return result

    def get_clone(self, flash_method, json_input, index, survey=None):
        """
        Iterate over a survey to create a modified clone from JSON input.

        :param flash_method: string for flash calculation name
        :param json_input: JSON input for flash calculation
        :param index: integer for index of incoming data
        :param survey: dictionary containing names and input values to modify in JSON

        :return clone: dictionary containing modified state variables and survey index
        """

        if survey is None:
            return json_input

        valid_flashes = [
            "wateranalysis",
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
        ]
        if flash_method not in valid_flashes:
            raise RuntimeError(
                "Invalid flash_method: {flash_method}. Use one of {', '.join(valid_flashes)}"
            )

        clone = deepcopy(json_input)
        for k, v in survey.items():
            if k.lower() != "recovery":
                d = clone["params"]
                if flash_method == "wateranalysis":
                    d = d["waterAnalysisInputs"]
                    for param in d:
                        if param["name"].lower() == k.lower():
                            if self.relative_inflows:
                                param["value"] += v[index]
                            else:
                                param["value"] = v[index]
                            _logger.debug(
                                f"Updating {k} for sample #{index+1} clone: new value = {param['value']}"
                            )
                else:
                    if k in d:
                        pass
                    elif k in d["inflows"]["values"]:
                        d = d["inflows"]["values"]
                    elif "corrosionParameters" in d:
                        if k in d["corrosionParameters"]:
                            d = d["corrosionParameters"]
                    else:
                        _logger.warning(f"Survey key {k} not found in JSON input keys: {json_input.keys()}.")
                    if self.relative_inflows:
                        if isinstance(d[k], dict):
                            d[k]["value"] += v[index]
                            val = d[k]["value"]
                        else:
                            d[k] += v[index]
                            val = d[k]
                    else:
                        if isinstance(d[k], dict):
                            d[k]["value"] = v[index]
                            val = d[k]["value"]
                        else:
                            d[k] = v[index]
                            val = d[k]
                    _logger.debug(
                        f"Updating {k} for sample #{index} clone: new value = {val}"
                    )

        recovery_key = None
        for k in survey:
            if k.lower() == "recovery":
                recovery_key = k
        if recovery_key:
            def _water_recovery(true_conc, pct_recovery):
                conc_factor = 1/(1-pct_recovery/100)
                return true_conc*conc_factor

            if flash_method == "wateranalysis":
                for param in clone["params"]["waterAnalysisInputs"]:
                    if "charge" in param:
                        param["value"] = _water_recovery(
                            param["value"],
                            survey[recovery_key][index],
                        )
            else:
                loc = clone["params"]["inflows"]["values"]
                for k,v in loc.items():
                    loc[k] = _water_recovery(v, survey[recovery_key][index])
        return clone

    def get_apparent_species_from_true(
        self,
        true_species_json,
        oliapi_instance,
        dbs_file_id,
        phase=None,
        file_name=None,
    ):
        """
        Run Water Analysis to get apparent species.

        :param true_species_json: JSON generated from true species
        :param oliapi_instance: instance of OLI Cloud API to call
        :param dbs_file_id: string ID of DBS file
        :param phase: string for inflows phase
        :param file_name: string for file to write, if any

        :return apparent_species: dictionary for molecular concentrations
        """

        stream_output = self.run_flash(
            "wateranalysis",
            oliapi_instance,
            dbs_file_id,
            true_species_json,
            burst_job_tag=None,
        )
        if phase is None:
            phase = "total"

        extracted_result = stream_output["result"][f"molecularConcentration_{phase}"]

        unit = input_unit_set["molecularConcentration"]["oli_unit"]
        concentrations = {k: v["values"][0] for k, v in extracted_result.items()}
        inflows = {"unit": unit, "values": concentrations}
        if file_name:
            write_output(inflows, file_name)
        return inflows
