// Copyright (c) 2018 Intel Corporation
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

#include "include/fetch.cl"
#include "include/mmad.cl"

#define SCALE 0.11f

#ifdef LIGHTWEIGHT_QUANTIZATION

#define QUANTIZATION \
    uchar4 out;\
    out[0] = convert_uchar_sat((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 0][b] * SCALE + bias_f.s0);\
    out[1] = convert_uchar_sat((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 1][b] * SCALE + bias_f.s1);\
    out[2] = convert_uchar_sat((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 2][b] * SCALE + bias_f.s2);\
    out[3] = convert_uchar_sat((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 0][b] * SCALE + bias_f.s3);

#elif NO_QUANTIZATION

#define QUANTIZATION \
    uchar4 out;\
    out[0] = convert_uchar_sat(dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 0][b]);\
    out[1] = convert_uchar_sat(dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 1][b]);\
    out[2] = convert_uchar_sat(dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 2][b]);\
    out[3] = convert_uchar_sat(dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 3][b]);

#else

#define QUANTIZATION \
    char4 out;\
    out[0] = ACTIVATION(convert_char(round( ((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 0][b]  * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0 ) ), ACTIVATION_PARAMS);\
    out[1] = ACTIVATION(convert_char(round( ((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 1][b]  * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1 ) ), ACTIVATION_PARAMS);\
    out[2] = ACTIVATION(convert_char(round( ((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 2][b]  * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2 ) ), ACTIVATION_PARAMS);\
    out[3] = ACTIVATION(convert_char(round( ((float)dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * 3][b]  * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3 ) ), ACTIVATION_PARAMS);

#endif

#define FILTER_IFM_MMAD_NUM ((FILTER_IFM_NUM + 31) / 32)
#define FILTER_OFM_MMAD_NUM ((FILTER_OFM_NUM + 7) / 8)
#define FILTER_IFM_ALIGNED (FILTER_IFM_MMAD_NUM * 32)
#define FILTER_OFM_ALIGNED (FILTER_OFM_MMAD_NUM * 8)
// input data is in blocks 4batch x 32 features

#define NEEDED_INPUT_X ((OUT_BLOCK_WIDTH-1) * (STRIDE_SIZE_X) + (FILTER_SIZE_X - 1) + 1)
#define NEEDED_INPUT_Y ((OUT_BLOCK_HEIGHT-1) * (STRIDE_SIZE_Y) + (FILTER_SIZE_Y - 1) + 1)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(convolution_mmad_batched_block_1x1)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
    __global BIAS_TYPE* biases,
    const __global float* quantizations,
#if CALIBRATION_TERM
    const __global float* calibrations,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0) * OUT_BLOCK_WIDTH;
    const uint y = get_global_id(1) * OUT_BLOCK_HEIGHT;

    const uint b_f = (get_group_id(2) * WG_BATCH_COUNT + get_sub_group_id());
#if WEIGHTS_PER_WORKITEM == 4
    const uint f = (b_f * 32 + get_sub_group_local_id() * 4) % FILTER_OFM_ALIGNED;
#else
    const uint f = ((b_f * WEIGHTS_PER_WORKITEM * 8) + get_sub_group_local_id() ) % FILTER_OFM_ALIGNED;
#endif
    const uint b_block = (b_f * 8 * WEIGHTS_PER_WORKITEM) / FILTER_OFM_ALIGNED;

    int4 dotProd[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT * WEIGHTS_PER_WORKITEM] = { 0 };

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = ((b_f * WEIGHTS_PER_WORKITEM) % FILTER_OFM_MMAD_NUM) * FILTER_OFM_BLOCK_PITCH;
    uint input_offset = IN_OFFSET + IN_B_BLOCK_PITCH * b_block;

    uint filter_idx = filter_offset;
    for (uint k = 0; k < FILTER_IFM_MMAD_NUM; ++k)
    {
        ////// preloading input data //////
        int4 preloaded_input[NEEDED_INPUT_X * NEEDED_INPUT_Y];
        for(int h = 0; h < NEEDED_INPUT_Y; h++)
        {
            for(int p = 0; p < NEEDED_INPUT_X; p++)
            {
                const int input_offset_y = input_y + h;
                const int input_offset_x = input_x + p;

                uint input_idx = input_offset + input_offset_y * IN_Y_PITCH + input_offset_x * IN_X_PITCH;
                preloaded_input[p + h * NEEDED_INPUT_X] = as_int4(intel_sub_group_block_read4((const __global uint*)(input + input_idx)));
            }
        }

        __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
        for (uint j = 0; j < FILTER_SIZE_Y; ++j)
        {
            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (uint i = 0; i < FILTER_SIZE_X; ++i)
            {
                ////// preloading weights data //////
                int8 preloaded_weights[WEIGHTS_PER_WORKITEM];
                __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
                for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
                {
                    preloaded_weights[w] = as_int8(intel_sub_group_block_read8((const __global uint*) (weights + (filter_idx + w * FILTER_OFM_BLOCK_PITCH) ) ));
                }

                ////// computing //////
                __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
                for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
                {
                    __attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
                    for(uint oy = 0; oy < OUT_BLOCK_HEIGHT; oy++)
                    {
                        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
                        for(uint ox = 0; ox < OUT_BLOCK_WIDTH; ox++)
                        {
                            const uint out_idx = ox + OUT_BLOCK_WIDTH * (oy + w * OUT_BLOCK_HEIGHT);
                            const uint preloaded_idx =ox * STRIDE_SIZE_X + i + NEEDED_INPUT_X * (oy * STRIDE_SIZE_Y + j);
                            dotProd[out_idx] = MMAD_4x8(preloaded_input[preloaded_idx], preloaded_weights[w], dotProd[out_idx]);
                        }
                    }
                }
                filter_idx += FILTER_X_PITCH;
            }
        }
        input_offset += IN_F_BLOCK_PITCH;
    }


#if WEIGHTS_PER_WORKITEM == 4

float4 quant_f = vload4(0, quantizations + f);
float4 bias_f = vload4(0, biases + f);
float4 calib_f = vload4(0, calibrations + f);

uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b_block*4, f, y, x);

__attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
{
    uint tmp_dst_index = dst_index;
    __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
    for(uint o = 0; o < OUT_BLOCK_WIDTH; o++)
    {
        uint4 to_output;
        __attribute__((opencl_unroll_hint(4)))
        for(uint b = 0; b < 4; b++)
        {
            const uint out_idx = o + OUT_BLOCK_WIDTH * h;

            QUANTIZATION;
            to_output[b] = as_uint(out);
        }
        intel_sub_group_block_write4((__global uint*)(output + tmp_dst_index), to_output);
        tmp_dst_index += OUT_X_PITCH;
    }
    dst_index += OUT_Y_PITCH;
}

#else // WEIGHTS_PER_WORKITEM ==4

////// QUANTIZE & OUTPUT //////
__attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
{
    float quant_f = quantizations[f + w * 8];
    float bias_f = biases[f + w * 8];
#if CALIBRATION_TERM
    float calib_f = calibrations[f + w * 8];
#endif
    __attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
    for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
    {
        __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
        for(uint o = 0; o < OUT_BLOCK_WIDTH; o++)
        {
            const uint out_idx = o + OUT_BLOCK_WIDTH * (h + w * OUT_BLOCK_HEIGHT);
            for(uint b = 0; b < 4; b++)
            {
            #if CALIBRATION_TERM
                dotProd[out_idx][b] = (UNIT_TYPE)round(((float)dotProd[out_idx][b] * quant_f * I_QF + bias_f) * calib_f);
            #else  // CALIBRATION_TERM
                dotProd[out_idx][b] = (UNIT_TYPE)round(((float)dotProd[out_idx][b] * quant_f * I_QF + bias_f) * O_QF);
            #endif // CALIBRATION_TERM
            }
        }
    }
}

////// OUTPUT STAGE //////
__attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
for(uint h = 0; h < OUT_BLOCK_HEIGHT; h++)
{
    __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
    for(uint o = 0; o < OUT_BLOCK_WIDTH; o++)
    {
        const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b_block*4, f, y + h, x + o);
        
        __attribute__((opencl_unroll_hint(4)))
        for(uint b = 0; b < 4; b++)
        {
            #if WEIGHTS_PER_WORKITEM == 2
                char2 out;
                const uint out_idx = o + OUT_BLOCK_WIDTH * h;
                out[0] = ACTIVATION(convert_char(dotProd[out_idx][b]), ACTIVATION_PARAMS);
                out[1] = ACTIVATION(convert_char(dotProd[out_idx + OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT][b]), ACTIVATION_PARAMS);

                intel_sub_group_block_write_uc2((__global uchar*)(output + dst_index + b * 32), as_uchar2(out));
            #else
            __attribute__((opencl_unroll_hint(WEIGHTS_PER_WORKITEM)))
            for(uint w = 0; w < WEIGHTS_PER_WORKITEM; w++)
            {
                const uint out_idx = o + OUT_BLOCK_WIDTH * (h + w * OUT_BLOCK_HEIGHT);
                const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b_block*4, f + w * 8, y + h, x + o);
                char char_val = ACTIVATION(convert_char(dotProd[out_idx][b]), ACTIVATION_PARAMS);
                output[dst_index + b * 32] = char_val;
            }
            #endif
        }
    }
}

#endif // WEIGHTS_PER_WORKITEM ==4

}

#undef FILTER_IFM_MMAD_NUM
#undef FILTER_OFM_MMAD_NUM
#undef FILTER_IFM_ALIGNED
#undef FILTER_OFM_ALIGNED


#undef SCALE
#undef QUANTIZATION