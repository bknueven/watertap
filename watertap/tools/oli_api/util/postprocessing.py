import logging
import json
import matplotlib.pyplot as plt
from itertools import product
import numpy as np

_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "OLIAPI - %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
)
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.INFO)

def _flatten_results(processed_requests):
    """
    Flatten OLI output.

    :param processed_requests: list of requests processed by OLIApi

    :return output_dict: flattened dictionary containing all inputs and outputs from OLI flash calls
    """

    _logger.info("Flattening OLI stream output ... ")

    props = []
    terminal_keys = ["unit", "value", "found", "fullVersion", "values"]

    def _find_props(data, path=None):
        """
        Get the path to all nested items in input data (recursive search).

        :param data: dictionary containing OLI flash output
        :param path: list of paths to endpoint

        :return props: list of nested path lists
        """
        path = path if path is not None else []
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (str, bool)):
                    props.append([*path, k])
                elif isinstance(v, list):
                    if all(k not in terminal_keys for k in v):
                        _find_props(v, [*path, k])
                elif isinstance(v, dict):
                    if all(k not in terminal_keys for k in v):
                        _find_props(v, [*path, k])
                    else:
                        props.append([*path, k])
        elif isinstance(data, list):
            for idx, v in enumerate(data):
                if isinstance(v, (dict, list)):
                    if all(k not in terminal_keys for k in v):
                        _find_props(v, [*path, idx])
                    else:
                        props.append([*path, idx])
        else:
            raise RuntimeError(f"Unexpected type for data: {type(data)}")

    def _get_nested_data(data, keys):
        for key in keys:
            data = data[key]
        return data

    def _extract_values(data, keys):
        values = _get_nested_data(data, keys)
        extracted_values = {}
        if isinstance(values, str):
            extracted_values = values
        elif isinstance(values, bool):
            extracted_values = bool(values)
        elif isinstance(values, dict):
            if any(k in values for k in ["group", "name", "fullVersion"]):
                if "value" in values:
                    extracted_values.update({"values": values["value"]})
                if "unit" in values:
                    unit = values["unit"] if values["unit"] else "dimensionless"
                    extracted_values.update({"units": unit})
                elif "fullVersion" in values:
                    extracted_values = {"fullVersion": values["fullVersion"]}
            elif all(k in values for k in ["found", "phase"]):
                extracted_values = values
            else:
                unit = values["unit"] if values["unit"] else "dimensionless"
                if "value" in values:
                    extracted_values = {
                        "units": unit,
                        "values": values["value"],
                    }
                else:
                    extracted_values = {
                        k: {
                            "units": unit,
                            "values": values["values"][k],
                        }
                        for k, v in values["values"].items()
                    }
        else:
            raise RuntimeError(f"Unexpected type for data: {type(values)}")
        return extracted_values

    def _create_input_dict(props, result):
        input_dict = {k: {} for k in set([prop[0] for prop in props])}
        for prop in props:
            k = prop[0]
            phase_tag = ""
            if "metaData" in prop:
                prop_tag = prop[-1]
            elif "result" in prop:
                # get property tag
                if isinstance(prop[-1], int):
                    prop_tag = prop[-2]
                else:
                    prop_tag = prop[-1]
                # get phase tag
                if any(k in prop for k in ["phases", "total"]):
                    if "total" in prop:
                        phase_tag = "total"
                    else:
                        phase_tag = prop[prop.index("phases") + 1]
            elif "submitted_requests" in prop:
                prop_tag = prop[-1]
                if "params" in prop:
                    if isinstance(prop[-1], int):
                        prop_tag = _get_nested_data(result, prop)["name"]
            else:
                _logger.warning(f"Unexpected result in result")
            label = f"{prop_tag}_{phase_tag}" if phase_tag else prop_tag
            input_dict[k][label] = _extract_values(result, prop)
        return input_dict

    float_nan = float("nan")

    def _add_to_output(input_dict, output_dict, index, number_samples):
        """
        Add incoming flash results to output data.

        :param input_dict: dictionary for incoming data
        :param output_dict: dictionary for output data
        :param index: integer for index of incoming data
        :param number_samples: integer for total number of incoming data samples
        """

        for k, v in input_dict.items():
            try:
                val = float(v)
            except:
                val = None
            if val is not None:
                if k not in output_dict:
                    output_dict[k] = [float_nan] * number_samples
                output_dict[k][index] = val
            elif isinstance(v, str):
                if k not in output_dict:
                    if k in ["fullVersion", "units"]:
                        output_dict[k] = v
                    else:
                        output_dict[k] = [float_nan] * number_samples
                if k in ["fullVersion", "units"]:
                    if input_dict[k] != output_dict[k]:
                        raise Exception(f"Input and output do not agree for key {k}")
                else:
                    output_dict[k][index] = v
            elif isinstance(v, dict):
                if k not in output_dict:
                    output_dict[k] = {}
                _add_to_output(input_dict[k], output_dict[k], index, number_samples)
            else:
                raise Exception(f"Unexpected value: {v}")

    output_dict = {}
    num_samples = len(processed_requests)
    for result in processed_requests:
        for idx, r in result.items():
            props = []
            _find_props(r)
            input_dict = _create_input_dict(props, r)
            _add_to_output(input_dict, output_dict, idx, num_samples)
    return output_dict

def heat_map(data, ax_title, col_labels, col_title, row_labels, row_title, cbar_label=None, ax=None, cbar_kw=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    im = ax.imshow(data, **kwargs)

    colorbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    colorbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    ax.set_title(ax_title)
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_xlabel(col_title)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.set_ylabel(row_title)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, va="top", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, colorbar




def scaling_tendency_plot(survey_inputs, output, cutoff=0.8):

    # post-process scaling data
    raw_st = output["result"]["prescalingTendencies"]
    cutoff_st = {}
    for k,v in raw_st.items():
        if max(raw_st[k]["values"]) >= cutoff and k != "H2O":
            cutoff_st[k] = [round(val, 2) for val in v["values"]]
    # post-process survey inputs
    set_inputs = {}
    for k,v in survey_inputs.items():
        labels = sorted(set(v))
        set_inputs[k] = [round(label, 1) for label in labels]

    if len(set_inputs) == 1:
        # plot ST vs dosant
        pass
        '''
        xs = [dosants[d] for d in dosants]
        ys = [scalants[s] for s in scalants]
        for idx, instance in enumerate(product(*[dosants, scalants])):
            print(idx, instance)

        for k,v in datasets.items():
            plt.figure()
            ys = v["ys"]
            xs = v["xs"]
            plt.title(f"{v['keys'][0]} vs. {v['keys'][1]}")
            plt.xlabel("Dosant Level")
            plt.ylabel("Scaling Tendency (-)")
            plt.axhline(y=1, color='r', linestyle='-')
            plt.axhline(y=cutoff, color='g', linestyle='--')
            plt.scatter(xs, ys, marker=".")
        '''
    elif len(set_inputs) == 2:
        # plot heatmap with color mapping for scaling tendency

        ks = list(set_inputs.keys())
        ncol = len(set_inputs[ks[0]])
        nrow = len(set_inputs[ks[1]])

        for k,v in cutoff_st.items():
            fig, ax = plt.subplots()
            heat_map(data=np.reshape(v, (nrow, ncol)),
                     ax_title=f"Scaling Tendency for {k}",
                     col_labels=set_inputs[ks[0]],
                     col_title=ks[0],
                     row_labels=set_inputs[ks[1]],
                     row_title=ks[1],
                     cbar_label="scaling tendency",
                     ax=ax, cmap="bwr", vmin=0, vmax=2)
    '''
    # map dosants to scaling tendencies
    elif len(dosants) == 0:
        # maybe plot ST vs sample #
        pass
    else:
        # too complex to plot normally
        pass
    '''
