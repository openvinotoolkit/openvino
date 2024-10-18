# Copyright (C) 2018-2024 Intel Corporation
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

nodes_to_compare = ("ScaledDotProductAttention", "PagedAttentionExtension", "Parameter", "ReadValue", "Assign")

def main():
    use_cache_eviction = False
    if len(sys.argv) >= 2:
        use_cache_eviction = sys.argv[1].lower() in 'true'

    OUTPUT_FILE = Path(os.path.join(os.path.dirname(__file__)), 'sdpa2pa_ref_diff' + ('_cache_eviction.txt' if use_cache_eviction else '.txt'))

    if OUTPUT_FILE.exists() and OUTPUT_FILE.is_file():
        OUTPUT_FILE.unlink()
    
    with open(OUTPUT_FILE, 'w') as file:
        model_list = utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit"))
        print(OUTPUT_FILE)
        print('ref_diff_map_cache_eviction = {' if use_cache_eviction else 'ref_diff_map = {', file=file)

        for model_id, _, _, _ in model_list:
            # wrapping in try/catch block to continue printing models even if one has failed
            try:
                model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)
            except:
                continue

            before_map = {}
            for op in model.model.get_ordered_ops():
                if op.get_type_name() in nodes_to_compare:
                    before_map[op.get_type_name()] = before_map.get(op.get_type_name(), 0) + 1

            # wrapping in try/catch block to continue printing models even if one has failed
            try:
                paged_attention_transformation(model.model, use_cache_eviction, use_cache_eviction)
            except:
                continue

            after_map = {}
            for op in model.model.get_ordered_ops():
                if op.get_type_name() in nodes_to_compare:
                    after_map[op.get_type_name()] = after_map.get(op.get_type_name(), 0) + 1

            print(f'\t"{model_id}" : {{', file=file)
            for op in set(after_map.keys()) | set(before_map.keys()):
                print(f'\t\t"{op}" : {after_map.get(op, 0) - before_map.get(op, 0)},', file=file)
            print('\t},', file=file)
        print('}', file=file)

if __name__ == "__main__":
    main()