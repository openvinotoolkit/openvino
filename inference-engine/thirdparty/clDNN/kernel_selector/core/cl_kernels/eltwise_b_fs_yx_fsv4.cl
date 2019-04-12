/*
// Copyright (c) 2019 Intel Corporation
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
*/

#include "include/include_all.cl"

#define PACK 4

#define SGR_MAX_SIZE   (get_max_sub_group_size())
#define SGR_LOCAL_ID   (get_sub_group_local_id())

#define GET_INDEX(_x) \
   ( ((_x / SGR_MAX_SIZE) * SGR_MAX_SIZE /* Normed to max_subgroup_size */)   \
     * (4 * sizeof(int)                  /* 4xINT32 per sub_group reading */) \
   )

inline int16 FUNC(get_int16)(const __global UNIT_TYPE* src, uint idx)
{
    int4 int_data = as_int4(intel_sub_group_block_read4((const __global uint*)(src + idx)));
    int16 to_return;
    for(uint i = 0; i < 4; i++)
    {
        for(uint j = 0; j < 4; j++)
        {
            to_return[i * 4 + j] = as_char4(int_data[i])[j];
        }
    }
    return to_return;
}
#define GET_INPUT(A, B) FUNC_CALL(get_int16)(A, GET_INDEX(x))


__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(eltwise_b_fs_yx_fsv4)(
    INPUTS_DECLS
    __global UNIT_TYPE* output
#if CALIBRATION_TERM
    , const __global float* calibrations
#endif
    )
{
    // This kernel works with linearized data w/o strides and padding
    // so only one dimension 'X' is required
    const uint x   = get_global_id(0);
    const uint idx = GET_INDEX(x);

    int16 res;

    DO_ELTWISE;

    for(uint i = 0; i < 4; i++)
    {
        const uint out_idx = idx + (sizeof(int) * (SGR_LOCAL_ID + (i * SGR_MAX_SIZE)));
        char4 char_res;

        for(uint j = 0; j < 4; j++)
        {
            int res_tmp = res[i * 4 + j];
        #if QUANTIZATION_TERM
        #if CALIBRATION_TERM
            // Batch:
            const uint b = out_idx / OUTPUT_BATCH_PITCH;
            // Feature:
            // Because of specific data layout Feature  must be normed to PACK size
            uint d3 = ((out_idx - b * OUTPUT_BATCH_PITCH) / (OUTPUT_FEATURE_PITCH * PACK)) * PACK;
            res_tmp = (int)round(((float)res_tmp) * calibrations[d3+j]);
        #else  // CALIBRATION_TERM
            res_tmp = (int)round(((float)res_tmp) * O_QF);
        #endif // CALIBRATION_TERM
        #endif // QUANTIZATION_TERM
        
        #ifdef ELTW_UNSIGNED
            char_res[j] = ACTIVATION(convert_uchar(res_tmp), NL_M, NL_N);
        #else
            char_res[j] = ACTIVATION(convert_char(res_tmp), NL_M, NL_N);
        #endif
        }
        // put 4 chars into output
        // char_result[i] = as_int(char_res);
        *((__global int*)(output + out_idx)) = as_int(char_res);
    }
}

#undef PACK
#undef SGR_MAX_SIZE
#undef SGR_LOCAL_ID
#undef GET_INDEX
#undef GET_INPUT
