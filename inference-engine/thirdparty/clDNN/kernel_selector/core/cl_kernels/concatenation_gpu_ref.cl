// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/include_all.cl"

#define GET_INDEX(prefix, ORDER) CAT(prefix, _GET_INDEX)(ORDER)

KERNEL (concatenation_gpu_ref)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint output_offset_in_concat_axis)
{
    const uint d1 = get_global_id(0); // Y
    const uint d2 = get_global_id(1); // F
#ifdef CHECK_FEATURES
    if (d2 >= INPUT0_FEATURE_NUM)
        return;
#endif
    const uint d3 = get_global_id(2); // B

    for (size_t d0 = 0; d0 < INPUT0_SIZES[INPUT_DIM_0]; ++d0) // X
    {
        uint input_offset = GET_INDEX(INPUT0, INPUT_DIMS_ORDER);
        uint output_offset = GET_INDEX(OUTPUT, OUTPUT_DIMS_ORDER);
        output[output_offset] = ACTIVATION(input[input_offset], ACTIVATION_PARAMS);
    }
}
