# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

'''
Use this script if you need to regenerate reference diffs for each model
to test SDPAToPA transformation.

The script will produce sdpa2pa_ref_diff.txt (or sdpa2pa_ref_diff_cache_eviction.txt
if using cache-eviction) containing a map in the
following format with nodes number changes for each model:

ref_diff_map = {
	"hf-internal-testing/tiny-random-LlamaForCausalLM" : {
		"PagedAttentionExtension" : 2,
		"ScaledDotProductAttention" : -2,
		"Parameter" : 7,
		"ReadValue" : -4,
		"Assign" : -4,
	},
	"hf-internal-testing/tiny-random-CohereForCausalLM" : {
		"PagedAttentionExtension" : 2,
		"ScaledDotProductAttention" : -2,
		"Parameter" : 7,
		"ReadValue" : -4,
		"Assign" : -4,
	},
    .
    .
    .
}

The map has to be pasted into sdpa2pa_ref_diff.py (same directory) for
includes to test SDPAToPA transformation.

Run the script by using 'python generate_ref_diffs.py' or 'python generate_ref_diffs.py True'
for generating the same map, but utilizing cache-eviction.
'''

import os
import sys
from pathlib import Path
import models_hub_common.utils as utils
from openvino._offline_transformations import paged_attention_transformation
from openvino._pyopenvino.op import _PagedAttentionExtension, Parameter, Result
from optimum.intel import OVModelForCausalLM
from optimum.intel.openvino import OVModelForVisualCausalLM
from typing import Type, Union

nodes_to_compare = ("ScaledDotProductAttention", "PagedAttentionExtension", "Parameter", "ReadValue", "Assign")

def get_models_list_type(file_name: str, cls: Union[Type[OVModelForCausalLM], Type[OVModelForVisualCausalLM]]):
    models = []
    for line_items in utils.parse_list_file(file_name):
        if len(line_items) == 2:
            model_name, model_link = line_items
            models.append((model_name, model_link, None, None, cls))
        elif len(line_items) == 4:
            model_name, model_link, mark, reason = line_items
            models.append((model_name, model_link, mark, reason, cls))
        elif len(line_items) > 4:
            model_name, model_link, mark, reason, *other = line_items
            if not mark:
                mark = None
            if not reason:
                reason = None
            other = line_items[4:]
            transformations = [item[8:] for item in other if item.startswith('ts_name:')]
            layers = [item[6:] for item in other if item.startswith('layer:')]
            models.append((model_name, model_link, mark, reason, transformations, layers))
        else:
            items = ','.join(line_items)
            assert False, \
                f'Incorrect model info fields {items}. It must contain either 2 or 4 or more than 4 fields.'
    return models

def main():
    use_cache_eviction = False
    if len(sys.argv) >= 2:
        use_cache_eviction = sys.argv[1].lower() in 'true'

    OUTPUT_FILE = Path(os.path.join(os.path.dirname(__file__)), 'sdpa2pa_ref_diff' + ('_cache_eviction.txt' if use_cache_eviction else '.txt'))

    if OUTPUT_FILE.exists() and OUTPUT_FILE.is_file():
        OUTPUT_FILE.unlink()

    with open(OUTPUT_FILE, 'w') as file:
        model_list = get_models_list_type(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit"), OVModelForCausalLM)
        model_list.extend(get_models_list_type(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-vl-models-precommit"), OVModelForVisualCausalLM))
        print(OUTPUT_FILE)
        print('ref_diff_map_cache_eviction = {' if use_cache_eviction else 'ref_diff_map = {', file=file)

        for model_id, _, _, _, cls in model_list:
            # wrapping in try/catch block to continue printing models even if one has failed
            try:
                model = cls.from_pretrained(model_id, export=True, trust_remote_code=True)
            except:
                print(f"Couldn't read {model_id}.")
                continue

            ov_model = model.model if cls is OVModelForCausalLM else model.lm_model

            before_map = {}
            for op in ov_model.get_ordered_ops():
                if op.get_type_name() in nodes_to_compare:
                    before_map[op.get_type_name()] = before_map.get(op.get_type_name(), 0) + 1

            # wrapping in try/catch block to continue printing models even if one has failed
            try:
                paged_attention_transformation(ov_model, use_cache_eviction, use_cache_eviction, use_cache_eviction)
            except:
                print(f"Couldn't run SDPAToPA transformation on {model_id} and generate diffs.")
                continue

            after_map = {}
            for op in ov_model.get_ordered_ops():
                if op.get_type_name() in nodes_to_compare:
                    after_map[op.get_type_name()] = after_map.get(op.get_type_name(), 0) + 1

            print(f'\t"{model_id}" : {{', file=file)
            for op in sorted(set(after_map.keys()) | set(before_map.keys())):
                print(f'\t\t"{op}" : {after_map.get(op, 0) - before_map.get(op, 0)},', file=file)
            print('\t},', file=file)
        print('}', file=file)

    print(f"output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
