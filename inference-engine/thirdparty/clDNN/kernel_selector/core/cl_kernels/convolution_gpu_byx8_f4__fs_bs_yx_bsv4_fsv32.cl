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

#define FILTER_IFM_SLICE_PITCH (32 * (FILTER_SIZE_X_SLICES * 8) * FILTER_SIZE_Y)
#define FILTER_OFM_SLICE_PITCH (FILTER_IFM_SLICE_PITCH * FILTER_IFM_SLICES)

#define OUT_BLOCK_BATCH 2
#define OUT_BLOCK_HEIGHT 2
#define WEIGHTS_PER_WORKITEM 4 // currently needs to be set to 4, check output stage and float4 on quantizations etc.

#define SCALE 0.11f

#ifdef LIGHTWEIGHT_QUANTIZATION

#define QUANTIZATION \
    out[w + pb * 4] = convert_uchar_sat((float)dotProd[w*OUT_BLOCK_HEIGHT*OUT_BLOCK_BATCH + h*OUT_BLOCK_BATCH + pb][i] * SCALE + bias_f[w]);

#elif NO_QUANTIZATION

#define QUANTIZATION \
    out[w + pb * 4] = convert_uchar_sat(dotProd[w*OUT_BLOCK_HEIGHT*OUT_BLOCK_BATCH + h*OUT_BLOCK_BATCH + pb][i]);

#else

#define QUANTIZATION \
    out[w + pb * 4] = as_uchar( ACTIVATION( convert_char( round( ( (float)dotProd[w*OUT_BLOCK_HEIGHT*OUT_BLOCK_BATCH + h*OUT_BLOCK_BATCH + pb][i] * quant_f[w] * I_QF + bias_f[w]) * calib_f[w])), ACTIVATION_PARAMS));

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

    const uint bf_id = ((uint)get_group_id(0) * WG_BATCH_SIZE + (uint)get_sub_group_id()) * 8 * WEIGHTS_PER_WORKITEM;

    const uint f = (bf_id) % OUTPUT_FEATURE_NUM;
    const uint b = OUT_BLOCK_BATCH * (bf_id / OUTPUT_FEATURE_NUM);

    int8 dotProd[OUT_BLOCK_BATCH * OUT_BLOCK_HEIGHT * WEIGHTS_PER_WORKITEM] =  { 0 };

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    uint filter_offset = (f/8)*FILTER_OFM_SLICE_PITCH;
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
                const uint filter_spatial_offset = 32 * (i*8 + (FILTER_SIZE_X_SLICES * 8) * j);
                
                int8 act_reg[OUT_BLOCK_HEIGHT*OUT_BLOCK_BATCH]; // activations for MMAD

                // preload batch data
                __attribute__((opencl_unroll_hint(OUT_BLOCK_BATCH)))
                for(uint pb = 0; pb < OUT_BLOCK_BATCH; pb++)
                {
                    // preload spatial data
                    __attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
                    for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
                    {
                        uint input_idx = GET_DATA_BYX8_F4_INDEX(INPUT0, b + pb, k * 4, input_offset_y + h * STRIDE_SIZE_Y, input_x + i * 8);
                        int2 _input_data_01 = as_int2(intel_sub_group_block_read2((__global uint*)(input + input_idx)));
                        int _input_data_2 = as_int(intel_sub_group_block_read((__global uint*)(input + input_idx + 8 * 8)));

                        act_reg[h * OUT_BLOCK_BATCH + pb][0] = _input_data_01[0];
                        act_reg[h * OUT_BLOCK_BATCH + pb][1] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 1);
                        act_reg[h * OUT_BLOCK_BATCH + pb][2] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 2);
                        act_reg[h * OUT_BLOCK_BATCH + pb][3] = intel_sub_group_shuffle_down(_input_data_01[0], _input_data_01[1], STRIDE_SIZE_X * 3);
                        act_reg[h * OUT_BLOCK_BATCH + pb][4] = _input_data_01[1];
                        act_reg[h * OUT_BLOCK_BATCH + pb][5] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 1);
                        act_reg[h * OUT_BLOCK_BATCH + pb][6] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 2);
                        act_reg[h * OUT_BLOCK_BATCH + pb][7] = intel_sub_group_shuffle_down(_input_data_01[1], _input_data_2, STRIDE_SIZE_X * 3);
                    }
                }

                uint filter_idx = filter_offset + filter_spatial_offset;

                // preload weights
                int8 _weights[WEIGHTS_PER_WORKITEM];
                __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
                for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++) // iterate over output feature channels for weights
                {
                    _weights[w] = as_int8(intel_sub_group_block_read8((__global uint*)(weights + filter_idx)));
                    filter_idx += FILTER_OFM_SLICE_PITCH;
                }

                __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
                for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++) // iterate over output feature channels for weights
                {
                    __attribute__((opencl_unroll_hint(OUT_BLOCK_BATCH)))
                    for(uint pb = 0; pb < OUT_BLOCK_BATCH; pb++)
                    {
                        __attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
                        for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
                        {
                            // MMAD on 8x WEIGHTS_PER_WORKITEM input channels elements for 8x outputs in WI
                            dotProd[w*OUT_BLOCK_HEIGHT*OUT_BLOCK_BATCH + h*OUT_BLOCK_BATCH + pb] = MMAD_8x8(act_reg[h * OUT_BLOCK_BATCH + pb], _weights[w], dotProd[w*OUT_BLOCK_HEIGHT*OUT_BLOCK_BATCH + h*OUT_BLOCK_BATCH + pb]);
                        }
                    }
                }
            }
        }
        filter_offset += FILTER_IFM_SLICE_PITCH;
    }


const uint sg_local_f = get_sub_group_local_id() * 4;
float4 quant_f = vload4(0, quantizations + f + sg_local_f);
float4 bias_f = vload4(0, biases + f + sg_local_f);
float4 calib_f = vload4(0, calibrations + f + sg_local_f);

__attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
{
    const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f, y + h, x);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < 8; i++)
    {

    #if WEIGHTS_PER_WORKITEM == 4
    
        uchar8 out;
        __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
        for(uint pb = 0; pb < OUT_BLOCK_BATCH; pb++)
        {
            for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
            {
                QUANTIZATION;
            }
        }
        intel_sub_group_block_write2((__global unsigned int*)(output + dst_index + 32 * 4 * i), as_uint2(out));
    
    #else
        #error NOT IMPLEMENTED
        __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
        for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
        {
        #if CALIBRATION_TERM
            dotProd[w*OUT_BLOCK_HEIGHT + h][i] = (UNIT_TYPE)round(((float)dotProd[w*OUT_BLOCK_HEIGHT + h][i] * quant_f[w] * I_QF + bias_f[w]) * calib_f[w]);
        #else  // CALIBRATION_TERM
            dotProd[w*OUT_BLOCK_HEIGHT + h][i] = (UNIT_TYPE)round(((float)dotProd[w*OUT_BLOCK_HEIGHT + h][i] * quant_f[w] * I_QF + bias_f[w]) * O_QF);
        #endif // CALIBRATION_TERM
            output[dst_index + 32 * 4 * i + 8 * w] = ACTIVATION(convert_char(dotProd[w*OUT_BLOCK_HEIGHT + h][i]), ACTIVATION_PARAMS);
        }
    
    #endif
    }
}

}

#undef OUT_BLOCK_HEIGHT
#undef WEIGHTS_PER_WORKITEM

#undef FILTER_SIZE_X_SLICES
#undef FILTER_IFM_SLICES

#undef FILTER_IFM_SLICE_PITCH
#undef FILTER_OFM_SLICE_PITCH

#undef SCALE
#undef QUANTIZATION
