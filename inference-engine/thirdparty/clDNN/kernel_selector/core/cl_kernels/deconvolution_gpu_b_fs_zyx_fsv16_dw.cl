/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "include/include_all.cl"
#include "include/unit_type.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define FEATURE_SLICE_SIZE 16
#if X_BLOCK_SIZE == 1
    #define BLOCK_TYPE UNIT_TYPE
    #define DST_VAR dst
#else
    #define BLOCK_TYPE CAT(UNIT_TYPE, X_BLOCK_SIZE)
    #define DST_VAR dst[x_block]
#endif

__attribute__((intel_reqd_sub_group_size(FEATURE_SLICE_SIZE))) // attr:no-format
KERNEL(deconvolution_gpu_b_fs_zyx_fsv16_dw)(
        const  __global INPUT0_TYPE *input,
        __global OUTPUT_TYPE *output,
        const __global FILTER_TYPE *weights,
#if BIAS_TERM
        const __global BIAS_TYPE *bias,
#endif
        uint split_idx)
{
    const uint zyx = (uint)get_global_id(0);
    const uint x = (zyx % (OUTPUT_SIZE_X / X_BLOCK_SIZE)) * X_BLOCK_SIZE;
#if INPUT0_LAYOUT_B_FS_YX_FSV16
    const uint y = zyx / (OUTPUT_SIZE_X / X_BLOCK_SIZE);
    const uint z = 0;
#else
    const uint zy = zyx / (OUTPUT_SIZE_X / X_BLOCK_SIZE);
    const uint y = zy % OUTPUT_SIZE_Y;
    const uint z = zy / OUTPUT_SIZE_Y;
#endif
    const uint f_block = get_group_id(1);
    const uint sglid = get_sub_group_local_id();
    const uint f = f_block * FEATURE_SLICE_SIZE + sglid;
    const uint b = (uint)get_global_id(2);

    const int input_x = x + PADDING_SIZE_X - (FILTER_SIZE_X - 1);
    const int input_y = y + PADDING_SIZE_Y - (FILTER_SIZE_Y - 1);
    const int input_z = z + PADDING_SIZE_Z - (FILTER_SIZE_Z - 1);

    // Input offset calculations:
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X +  INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_z_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y +  INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_fs_pitch = input_z_pitch * (INPUT0_PAD_BEFORE_SIZE_Z +  INPUT0_SIZE_Z + INPUT0_PAD_AFTER_SIZE_Z);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              INPUT0_PAD_BEFORE_SIZE_Z * input_z_pitch +
                              INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
                              INPUT0_PAD_BEFORE_SIZE_X * input_x_pitch +
                              (f_block + input_fs_pad_before) * input_fs_pitch;

    const uint filter_offset = f_block * FEATURE_SLICE_SIZE * FILTER_SIZE_X * FILTER_SIZE_Y * FILTER_SIZE_Z;

#if BIAS_TERM
    BLOCK_TYPE dst = (BLOCK_TYPE)(UNIT_BLOCK_READ(bias, f_block * FEATURE_SLICE_SIZE));
#else
    BLOCK_TYPE dst = (BLOCK_TYPE)(UNIT_VAL_ZERO);
#endif

    UNIT_TYPE wei[FILTER_SIZE_Z * FILTER_SIZE_Y * FILTER_SIZE_X];

    unroll_for (uint k_z = 0; k_z < FILTER_SIZE_Z; k_z++) {
        unroll_for (uint k_y = 0; k_y < FILTER_SIZE_Y; k_y++) {
            unroll_for (uint k_x = 0; k_x < FILTER_SIZE_X; k_x++) {
                const uint wei_idx = (FILTER_SIZE_Z - k_z - 1) * FILTER_Z_PITCH + (FILTER_SIZE_Y - k_y - 1) * FILTER_Y_PITCH + (FILTER_SIZE_X - k_x - 1);
                wei[wei_idx] = UNIT_BLOCK_READ(weights, filter_offset + k_z * FILTER_Z_PITCH * FEATURE_SLICE_SIZE
                                                                      + k_y * FILTER_Y_PITCH * FEATURE_SLICE_SIZE
                                                                      + k_x * FEATURE_SLICE_SIZE);
            }
        }
    }

    UNIT_TYPE src_val = UNIT_VAL_ZERO;

    unroll_for (uint x_block = 0; x_block < X_BLOCK_SIZE; x_block++) {
        unroll_for (uint k_z = 0; k_z < FILTER_SIZE_Z; k_z++) {
            const int input_offset_z = input_z + k_z;
            const bool zero_z = (input_offset_z >= INPUT0_SIZE_Z * STRIDE_SIZE_Z) || (input_offset_z < 0) || ((input_offset_z % STRIDE_SIZE_Z) != 0);
            unroll_for (uint k_y = 0; k_y < FILTER_SIZE_Y; k_y++) {
                const int input_offset_y = input_y + k_y;
                const bool zero_y = (input_offset_y >= INPUT0_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);
                unroll_for (uint k_x = 0; k_x < FILTER_SIZE_X; k_x++) {
                    const int input_offset_x = input_x + k_x + x_block;
                    const bool zero_x = (input_offset_x >= INPUT0_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);
                    const uint in_idx = k_z * FILTER_Z_PITCH + k_y * FILTER_Y_PITCH + k_x;
                    if (!zero_z && !zero_y && !zero_x) {
                        uint fixed_input_offset_x = (uint)input_offset_x / STRIDE_SIZE_X;
                        uint fixed_input_offset_y = (uint)input_offset_y / STRIDE_SIZE_Y;
                        uint fixed_input_offset_z = (uint)input_offset_z / STRIDE_SIZE_Z;

                        src_val = UNIT_BLOCK_READ(input, input_offset +
                                                         fixed_input_offset_z * input_z_pitch +
                                                         fixed_input_offset_y * input_y_pitch +
                                                         fixed_input_offset_x * input_x_pitch);
                        DST_VAR = mad(src_val, wei[in_idx], DST_VAR);
                    }
                }
            }
        }
    }

    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X +  OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_z_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y +  OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_z_pitch * (OUTPUT_PAD_BEFORE_SIZE_Z +  OUTPUT_SIZE_Z + OUTPUT_PAD_AFTER_SIZE_Z);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset =  b * output_b_pitch +
                                (f_block + output_fs_pad_before) * output_fs_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_Z + z) * output_z_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_Y + y) * output_y_pitch +
                                (OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

#if OUTPUT_LEFTOVERS
    if ((f_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM)
    {
        unroll_for (uint x_block = 0; x_block < X_BLOCK_SIZE; x_block++) {
            if (f_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM)
                output[output_offset + (x + x_block) * output_x_pitch + sglid] = DST_VAR;
        }
    }
    else
#endif //  OUTPUT_LEFTOVERS
    {
        unroll_for (uint x_block = 0; x_block < X_BLOCK_SIZE; x_block++) {
            UNIT_BLOCK_WRITE(output, output_offset + (x + x_block) * output_x_pitch, DST_VAR);
        }
    }
}
