// Copyright (c) 2016-2017 Intel Corporation
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

#include "include/common.cl"

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

#define FILTER_IFM_SLICES ((FILTER_IFM_NUM + 3) /4)
#define FILTER_SIZE_X_SLICES ((FILTER_SIZE_X + 7) / 8)

#define OUT_BLOCK_HEIGHT 4
#define WEIGHTS_PER_WORKITEM 4 // currently needs to be set to 4, check output stage and float4 on quantizations etc.

#define SCALE 0.11f

#ifdef LIGHTWEIGHT_QUANTIZATION

#define QUANTIZATION \
    out[w] = convert_uchar_sat((float)dotProd[w*OUT_BLOCK_HEIGHT + h][i] * SCALE + bias_f[w]);

#elif NO_QUANTIZATION

#define QUANTIZATION \
    out[w] = convert_uchar_sat(dotProd[w*OUT_BLOCK_HEIGHT + h][i]);

#else

#define QUANTIZATION \
    out[w] = as_uchar( ACTIVATION( convert_char( round( ( (float)dotProd[w*OUT_BLOCK_HEIGHT + h][i] * quant_f[w] * I_QF + bias_f[w]) * calib_f[w])), NL_M, NL_N));

#endif

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(convolution_gpu_byx8_f4_fs_bs_yx_bsv4_fsv32)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
    __global BIAS_TYPE* biases,
    __global float* quantizations,
#if CALIBRATION_TERM
    __global float* calibrations,
#endif
    uint split_idx)
{
    const uint x = get_group_id(1) * 8;
    const uint y = get_group_id(2) * OUT_BLOCK_HEIGHT;

    const uint f = (get_group_id(0) * 8 * WEIGHTS_PER_WORKITEM ) % OUTPUT_FEATURE_NUM;
    const uint b = (get_group_id(0) * 8 * WEIGHTS_PER_WORKITEM) / OUTPUT_FEATURE_NUM;

    int8 dotProd[OUT_BLOCK_HEIGHT * WEIGHTS_PER_WORKITEM] =  { 0 };

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET;

    for (uint k = 0; k < FILTER_IFM_SLICES; ++k)
    {
        __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;

            __attribute__((opencl_unroll_hint(FILTER_SIZE_X_SLICES)))
            for(uint i = 0; i < FILTER_SIZE_X_SLICES; i++)
            {
                int8 act_reg[OUT_BLOCK_HEIGHT]; // activations for MMAD

                // preload spatial data
                __attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
                for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
                {
                    uint input_idx = GET_DATA_BYX8_F4_INDEX(INPUT0, b, k * 4, input_offset_y + h * STRIDE_SIZE_Y, input_x + i * 8);
                    int2 _input_data_01 = as_int2(intel_sub_group_block_read2((__global uint*)(input + input_idx)));
                    int _input_data_2 = as_int(intel_sub_group_block_read((__global uint*)(input + input_idx + 8 * 8)));

                    act_reg[h][0] = _input_data_01[0];
                    act_reg[h][1] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 1);
                    act_reg[h][2] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 2);
                    act_reg[h][3] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 3);
                    act_reg[h][4] = _input_data_01[1];
                    act_reg[h][5] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 1);
                    act_reg[h][6] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 2);
                    act_reg[h][7] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 3);
                }

                __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
                for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++) // iterate over output feature channels for weights
                {
                    uint filter_idx = GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(FILTER, f + w * 8, k * 4, j, i * 8);
                    int8 _w = as_int8(intel_sub_group_block_read8((__global uint*)(weights + filter_idx)));

                    __attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
                    for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
                    {
                        // MMAD on 8x WEIGHTS_PER_WORKITEM input channels elements for 8x outputs in WI
                        dotProd[w*OUT_BLOCK_HEIGHT + h] = MMAD_8x8(act_reg[h], _w, dotProd[w*OUT_BLOCK_HEIGHT + h]);
                    }
                }
            }
        }
    }

float4 quant_f = as_float4(intel_sub_group_block_read4((__global uint*) (quantizations + f) ));
float4 bias_f = as_float4(intel_sub_group_block_read4((__global uint*) (biases + f) ));
#if CALIBRATION_TERM
float4 calib_f = as_float4(intel_sub_group_block_read4((__global uint*) (calibrations + f) ));
#endif

__attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
{
    const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f + get_sub_group_local_id(), y + h, x);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < 8; i++)
    {

    #if WEIGHTS_PER_WORKITEM == 4
    
        uchar4 out;
        __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
        for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
        {
            QUANTIZATION;
        }
        intel_sub_group_block_write_uc4((__global uchar*)(output + dst_index + 32 * 4 * i), out);
    
    #else
    
        __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
        for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
        {
        #if CALIBRATION_TERM
            dotProd[w*OUT_BLOCK_HEIGHT + h][i] = (UNIT_TYPE)round(((float)dotProd[w*OUT_BLOCK_HEIGHT + h][i] * quant_f[w] * I_QF + bias_f[w]) * calib_f[w]);
        #else  // CALIBRATION_TERM
            dotProd[w*OUT_BLOCK_HEIGHT + h][i] = (UNIT_TYPE)round(((float)dotProd[w*OUT_BLOCK_HEIGHT + h][i] * quant_f[w] * I_QF + bias_f[w]) * O_QF);
        #endif // CALIBRATION_TERM
            output[dst_index + 32 * 4 * i + 8 * w] = ACTIVATION(convert_char(dotProd[w*OUT_BLOCK_HEIGHT + h][i]), NL_M, NL_N);
        }
    
    #endif
    }
}

}

#undef OUT_BLOCK_HEIGHT
#undef WEIGHTS_PER_WORKITEM

#undef FILTER_SIZE_X_SLICES
#undef FILTER_IFM_SLICES

#undef SCALE
#undef QUANTIZATION