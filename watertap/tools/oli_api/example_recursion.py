import json
import numpy as np
import sys


def get_input_from_output(output_dict, input_dict, index):
    for k, v in output_dict.items():
        if isinstance(v, (np.ndarray, list, tuple)):
            val = v[index]
            if not np.isnan(val):
                input_dict[k] = val
        elif isinstance(v, str):
            input_dict[k] = v
        elif isinstance(v, dict):
            input_dict[k] = {}
            get_input_from_output(output_dict[k], input_dict[k], index)
        else:
            raise Exception(f"unexpected value: {v}")


def add_to_output(input_dict, output_dict, index, number_samples):
    for k, v in input_dict.items():
        try:
            val = float(v)
        except:
            val = None 
        if val is not None:
            if k not in output_dict:
                output_dict[k] = np.full(number_samples, np.nan)
            output_dict[k][index] = val
        elif isinstance(v, str):
            if k not in output_dict:
                output_dict[k] = v
            if input_dict[k] != output_dict[k]:
                raise Exception(f"input and output strings do not agree for key {k}")
        elif isinstance(v, dict):
            if k not in output_dict:
                output_dict[k] = {}
            add_to_output(input_dict[k], output_dict[k], index, number_samples)
        else:
            raise Exception(f"unexpected value: {v}")


if __name__ == "__main__":

    files = sys.argv[1:]

    output_dict = {}
    number_samples = len(files)
    input_to_output_index = {}
    for index, f in enumerate(files):
        input_dict = json.load(open(f, "r"))
        add_to_output(input_dict, output_dict, index, number_samples)
        input_to_output_index[f] = index

    print(output_dict)
    print(input_to_output_index)

    input1 = {}
    get_input_from_output(output_dict, input1, input_to_output_index[files[0]])
    print(input1)

    input2 = {}
    get_input_from_output(output_dict, input2, input_to_output_index[files[1]])
    print(input2)
