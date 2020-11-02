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


#include "include/reshape_dims.cl"
#include "include/fetch.cl"

#include "include/data_types.cl"

#if !INPUT0_LAYOUT_BFYX && !INPUT0_LAYOUT_B_FS_YX_32FP
#error "Data binary reorder: unsupported input layout"
#endif

#if !OUTPUT_LAYOUT_BFYX && !OUTPUT_LAYOUT_B_FS_YX_32FP
#error "Data binary reorder: unsupported output layout"
#endif

#ifdef MEAN_SUBTRACT_IN_BUFFER
#error "Mean subtruction is not supported in binary reorder"
#endif


KERNEL (reorder_data_binary)(const __global INPUT_REORDER_TYPE* input,
                                   __global OUTPUT_REORDER_TYPE* output)
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    const uint y = ((uint)(get_global_id(2))) / INPUT0_SIZE_X;
    const uint x = ((uint)(get_global_id(2))) % INPUT0_SIZE_X;


#if BINARY_INPUT && BINARY_OUTPUT
    int input_index = INPUT0_OFFSET
                    + b * INPUT_PACKED_FEATURES_NUM * INPUT0_FEATURE_PITCH
                    + f * INPUT0_FEATURE_PITCH
                    + y * INPUT0_Y_PITCH
                    + x * INPUT0_X_PITCH;
    int output_index = OUTPUT_OFFSET
                     + b * OUTPUT_PACKED_FEATURES_NUM * OUTPUT_FEATURE_PITCH
                     + f * OUTPUT_FEATURE_PITCH
                     + y * OUTPUT_Y_PITCH
                     + x * OUTPUT_X_PITCH;

    output[output_index] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(input[input_index]), NL_M, NL_N);
#elif BINARY_OUTPUT
    int output_index = OUTPUT_OFFSET
                     + b * OUTPUT_PACKED_FEATURES_NUM * OUTPUT_FEATURE_PITCH
                     + f * OUTPUT_FEATURE_PITCH
                     + y * OUTPUT_Y_PITCH
                     + x * OUTPUT_X_PITCH;

    OUTPUT_TYPE res = 0x00000000;
    int limit = min((int)IFM_PACK_SIZE, (int)(INPUT0_FEATURE_NUM - f*IFM_PACK_SIZE));
    for (int c = 0; c < limit; c++)
    {
        // index of required bit
        int input_index = INPUT0_OFFSET
                        + b * INPUT0_BATCH_PITCH
                        + (f * IFM_PACK_SIZE + c) * INPUT0_FEATURE_PITCH
                        + y * INPUT0_Y_PITCH
                        + x * INPUT0_X_PITCH;

        int bit = input[input_index] > UNIT_VAL_ZERO ? 1 : 0;
        res |= (bit << c);
    }
    output[output_index] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(res), NL_M, NL_N);
#elif BINARY_INPUT
    int input_index = INPUT0_OFFSET
                    + b * INPUT_PACKED_FEATURES_NUM * INPUT0_FEATURE_PITCH
                    + f * INPUT0_FEATURE_PITCH
                    + y * INPUT0_Y_PITCH
                    + x * INPUT0_X_PITCH;
    int res = input[input_index];
    int limit = min((int)IFM_PACK_SIZE, (int)(INPUT0_FEATURE_NUM - f*IFM_PACK_SIZE));
    for (int c = 0; c < limit; c++)
    {
        int output_index = OUTPUT_OFFSET
                         + b * OUTPUT_BATCH_PITCH
                         + (f*IFM_PACK_SIZE + c) * OUTPUT_FEATURE_PITCH
                         + y * OUTPUT_Y_PITCH
                         + x * OUTPUT_X_PITCH;

        int bit = (res >> c) & 0x00000001 > 0 ? 1 : -1;
        output[output_index] = ACTIVATION_FUNC_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE(bit), NL_M, NL_N);
    }
#else
#error "Binary reorder is used without binary tensors"
#endif

}
