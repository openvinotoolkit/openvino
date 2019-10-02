// Copyright (c) 2018-2019 Intel Corporation
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
#include "include/fetch.cl"
#include "include/imad.cl"
#if QUANTIZATION_TERM || CALIBRATION_TERM || defined(O_QF)
#    define ACCUMULATOR_TYPE int
#    define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#    define ACTIVATION_TYPE float
#    define ACTIVATION_TYPE_BASE float
     // TODO: It's unclear for now what should be the rounding scheme. Might
     // happen that we will be required to round to nearest-even. Should it
     // become a customization point, or will we support just one rounding
     // scheme?
#    if OUTPUT_IS_FP
         // [U]INT8 -> float convolution with quantization/calibration.
#        define AFTER_CALIBRATION_ROUND(x) (x)
#    else
         // TODO: Do we need the round of the conv result in the fused
         // primitive?
#        define AFTER_CALIBRATION_ROUND(x) round(x)
#    endif
#elif defined(ACTIVATION_ELTW_TYPED) && (INT8_UNIT_USED || UINT8_UNIT_USED)
    // TODO: Get rid of INT8_UNIT_USED - its meaning doesn't look to be defined
    // properly. Better to check INPUT0_TYPE is [U]INT8 somehow...
#    define ACCUMULATOR_TYPE int
#    define TO_ACCUMULATOR_TYPE(x) convert_int(x)
#    define ACTIVATION_TYPE int
#    define ACTIVATION_TYPE_BASE int
#    define AFTER_CALIBRATION_ROUND(x) (x)
#else
#    define ACCUMULATOR_TYPE INPUT0_TYPE
#    define TO_ACCUMULATOR_TYPE(x) TO_INPUT0_TYPE(x)
#    define ACTIVATION_TYPE INPUT0_TYPE
#    define ACTIVATION_TYPE_BASE INPUT0
#    define AFTER_CALIBRATION_ROUND(x) (x)
#endif

#if defined(ACTIVATION_ELTW)
#    if OUTPUT_IS_FP
#        define AFTER_ELTW_CALIBRATION_ROUND(x) (x)
#    else
#        define AFTER_ELTW_CALIBRATION_ROUND(x) round(x)
#    endif
#endif

#if QUANTIZATION_TERM && (!defined(CALIBRATION_TERM) || CALIBRATION_TERM == 0 ) && !defined(O_QF)
    // To get proper type for the "before_activation" below.
#   define O_QF_LOCAL_DEFINE
#   define O_QF 1.0
#endif
#if NON_BLOCK_LOAD != 1
// block loads for inputs and weights should be fastest, but compiler seems
// to do better with a mix, regular loads for inputs and block loads for weights. 
#define BLOCK_LOAD_WEIGHTS
#endif
// Input reading operation is always blocked.
#define BLOCK_LOAD_INPUTS

// for now kernel stride is square
#define K_WSTRIDE K_STRIDE
#define K_HSTRIDE K_STRIDE

// need KERNEL width for first output + STRIDE more for each additional.
#define IN_BLOCK_WIDTH  (K_WIDTH  + K_WSTRIDE * (OUT_BLOCK_WIDTH  - 1))
#define IN_BLOCK_HEIGHT (K_HEIGHT + K_HSTRIDE * (OUT_BLOCK_HEIGHT - 1))

// for imad we are packing 4 8bit activations per 32 bit SIMD lane
// if we later add 4bit, then PACK would be 8.
#define PACK 4

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x) AS_TYPE_N(INPUT0_TYPE, 4, x)

// int8 conv_input and weights data is packed to int32 "batches",
// int/uint pointers here instead of INPUT0_TYPE/FILTER_TYPE for convenience
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL (fused_convolution_eltwise_gpu_imad)(
    const __global uint          *conv_input,
    __global OUTPUT_TYPE         *output,
    const __global int           *weights
#if BIAS_TERM
    , const __global BIAS_TYPE   *biases
#endif
#if QUANTIZATION_TERM
    , const __global float       *quantizations
#endif
#if CALIBRATION_TERM
    , const __global float       *calibrations
#endif
    , uint split_idx
// one kernel for both convolution and fused_conv_eltwise
#ifdef ACTIVATION_ELTW //defined for fused conv+eltwise
#    if IN_OUT_OPT
    // The argument is always present (in this case it would be the same as
    // output), just ignore it.
    , const __global OUTPUT_TYPE *_ignore
#    else
    , const __global INPUT1_TYPE *eltw_input
#    endif
#    if ELTW_CALIBRATION_TERM
    , const __global float* eltw_output_calibrations
#    endif
#endif
)
{
    const uint oc = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column
    const uint or = get_global_id(1) * OUT_BLOCK_HEIGHT; // or = Output Row
    const uint fm = get_global_id(2);                    // fm = Feature Map = od = Output Depth, SIMD is across this dimension, WG is 1x1x16
    const uint fmg = get_group_id(2);
    const uint lid = get_local_id(2);
    const uint batch = fm / _OD;
    const uint f = fm % _OD;

    uint in[IN_BLOCK_HEIGHT];
    int  out[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT] = { 0 };  // this is the 32 bit signed accumulator that must be converted to 8 bits before final write.

    #define NUM_FILTERS (K_HEIGHT * K_WIDTH)
    int w[NUM_FILTERS];

    int in_addr;

#ifdef BLOCK_LOAD_WEIGHTS
    int weight_addr = (fmg % ((_OD + SIMD_SIZE - 1) / SIMD_SIZE)) * ((_ID * K_HEIGHT * K_WIDTH * SIMD_SIZE) / PACK);
#else
    int weight_addr = (fmg % ((_OD + SIMD_SIZE - 1) / SIMD_SIZE)) * ((_ID * K_HEIGHT * K_WIDTH * SIMD_SIZE) / PACK) + lid;
#endif

    uint input_size = (_ID * (_IH + IHPAD) * (_IW + IWPAD)) / PACK; // dividing by PACK to get right number of 32bit entities.

    // For imad we do 4X less input feature map iterations since we are packing 4 of them in each uchar4.
    // _ID provided by host is multiple of packing factor.
    __attribute__((opencl_unroll_hint(1)))
    for(int kd = 0; kd < (_ID / PACK); kd++)
    {

#ifdef BLOCK_LOAD_INPUTS
        in_addr = kd * (_IH + IHPAD) * (_IW + IWPAD) + (or * K_STRIDE) * (_IW + IWPAD) + (oc * K_STRIDE);
#else
        in_addr = kd * (_IH + IHPAD) * (_IW + IWPAD) + (or * K_STRIDE) * (_IW + IWPAD) + (oc * K_STRIDE) + lid;
#endif
        in_addr += batch * input_size;  // adjust for batching

        for(uint reg = 0; reg < IN_BLOCK_HEIGHT; reg++) {
#ifdef BLOCK_LOAD_INPUTS
            in[reg] = intel_sub_group_block_read(&conv_input[in_addr]);
#else
            in[reg] = conv_input[in_addr];// read SIMD_SIZE elements wide
#endif
            in_addr += (_IW + IWPAD);  // move to next row down
        }

#ifdef BLOCK_LOAD_WEIGHTS
        *((int8*)&w[0]) = as_int8(intel_sub_group_block_read8((const __global uint*) &weights[weight_addr]));
        w[8]= as_int(intel_sub_group_block_read((const __global uint*) &weights[weight_addr + (SIMD_SIZE<<3)]));
        weight_addr += SIMD_SIZE*NUM_FILTERS;
#else
        for(int pf=0; pf < NUM_FILTERS; pf++) {
            w[pf] = weights[weight_addr];
            weight_addr += SIMD_SIZE;
        }
#endif

        int wi = 0;
        // This loop is temporarily not unrolled because the unroll causes TeamCity hangs.
        //__attribute__((opencl_unroll_hint(K_HEIGHT)))
        for (int kr = 0; kr < K_HEIGHT; ++kr) // kr = Kernel Row
        {
            __attribute__((opencl_unroll_hint(K_WIDTH)))
            for (int kc = 0; kc < K_WIDTH; ++kc) // kc = Kernel Column
            {
                for (int br = 0; br < OUT_BLOCK_HEIGHT; br++) {
                    for (int bc = 0; bc < OUT_BLOCK_WIDTH; bc++) {
                        uint input = sub_group_broadcast(in[br * K_HSTRIDE + kr], bc * K_WSTRIDE + kc);

                        out[br * OUT_BLOCK_WIDTH + bc] = TO_ACCUMULATOR_TYPE(IMAD(out[br * OUT_BLOCK_WIDTH + bc], AS_INPUT0_TYPE_4(input), as_char4(w[wi])));
                    }
                }
                wi++;
            }
        }
    } //for kd

    // Compiler emits worse code when GET_DATA_B_FS_YX_FSV4_INDEX is called inside the loop
    // to calculate out_idx and eltw_idx. Calculate offsets with GET_DATA_B_FS_YX_FSV4_INDEX before
    // entering the loop, and have a simple expressions for indexes inside the loop.
    const uint output_idx_offset = GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, batch, f, or, oc);
    const uint output_row_size_bytes = (_OW + OWPAD) * PACK;
#ifdef ACTIVATION_ELTW //defined for fused conv+eltwise
    #if IN_OUT_OPT == 0
    const uint eltw_idx_offset = GET_DATA_B_FS_YX_FSV4_INDEX(INPUT1, batch, f, or * ELTW_STRIDE_Y, oc * ELTW_STRIDE_X);
    const uint eltw_row_size_bytes = (INPUT1_SIZE_X + INPUT1_PAD_BEFORE_SIZE_X + INPUT1_PAD_AFTER_SIZE_X) * PACK;
    #endif
#endif

    for (int r = 0; r < OUT_BLOCK_HEIGHT; r++)
    {
        #if NEED_TO_VERIFY_OUTPUT_RANGES == 1
        const bool zero_r = or + r >= OUTPUT_SIZE_Y;
        if(!zero_r)
        #endif
        {
        for (int c = 0; c < OUT_BLOCK_WIDTH; c++)
        {
            #if NEED_TO_VERIFY_OUTPUT_RANGES == 1
            const bool zero_c = oc + c >= OUTPUT_SIZE_X;
            if(!zero_c)
            #endif
            {
            #if OUTPUT_LAYOUT_BYXF_AF32 == 1
                uint out_idx = GET_DATA_BYXF_AF32_INDEX(OUTPUT, batch, fm, or + r, oc + c);
            #elif OUTPUT_LAYOUT_B_FS_YX_FSV4 == 1
                uint out_idx = output_idx_offset + r * output_row_size_bytes + (c*PACK);
            #else
                #error "Incorrect output layout"
            #endif
            ACCUMULATOR_TYPE dotProd = out[r * OUT_BLOCK_WIDTH + c];

#if BIAS_TERM
    #if BIAS_PER_OUTPUT
            const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
    #elif BIAS_PER_OFM
            const uint bias_index = f;
    #endif
    #if !DONT_DEQUANTIZE_BIAS
        dotProd += TO_ACCUMULATOR_TYPE(biases[bias_index]);
    #endif
#endif
         ACTIVATION_TYPE dequantized = dotProd
    #if QUANTIZATION_TERM
                        * quantizations[f] * I_QF
    #endif
            ;
#if DONT_DEQUANTIZE_BIAS
    #if !BIAS_TERM || !QUANTIZATION_TERM
        #error "DONT_DEQUANTIZE_BIAS is meaningless without BIAS_TERM and QUANTIZATION_TERM"
    #endif
            dequantized += biases[bias_index];
#endif
            ACTIVATION_TYPE activated = ACTIVATION_CONV_TYPED(ACTIVATION_TYPE, dequantized, ACTIVATION_PARAMS_CONV_TYPED);
#if CALIBRATION_TERM
            activated = activated * calibrations[f];
#elif defined(O_QF)
            activated = activated * O_QF;
#endif
            activated = AFTER_CALIBRATION_ROUND(activated);

#if !defined ACTIVATION_ELTW //defined for fused conv+eltwise
            output[out_idx] = TO_OUTPUT_TYPE_SAT(activated);
#else
#   if IN_OUT_OPT == 0
    const uint eltw_idx_offset = GET_DATA_B_FS_YX_FSV4_INDEX(INPUT1, batch, f, or * ELTW_STRIDE_Y, oc * ELTW_STRIDE_X);
    const uint eltw_row_size_bytes = (INPUT1_SIZE_X + INPUT1_PAD_BEFORE_SIZE_X + INPUT1_PAD_AFTER_SIZE_X) * PACK;

    uint eltw_idx = eltw_idx_offset + r * ELTW_STRIDE_Y * eltw_row_size_bytes + (c * ELTW_STRIDE_X * PACK);

    INPUT1_TYPE eltw_elem = eltw_input[eltw_idx];
#   else
    uint eltw_idx = out_idx;
    OUTPUT_TYPE eltw_elem = output[eltw_idx];
#   endif

#    if defined(NON_CONV_SCALE)
    float eltw_elem_scaled = (float)eltw_elem * NON_CONV_SCALE;
#    else
    ACTIVATION_TYPE eltw_elem_scaled = CAT(CAT(TO_, ACTIVATION_TYPE_BASE), _TYPE_SAT)(eltw_elem);
#    endif
    ACTIVATION_TYPE eltw_out = eltw_elem_scaled + activated;

    eltw_out = ACTIVATION_ELTW_TYPED(ACTIVATION_TYPE, eltw_out, ACTIVATION_PARAMS_ELTW_TYPED);

    eltw_out = AFTER_ELTW_CALIBRATION_ROUND(eltw_out
#    if ELTW_CALIBRATION_TERM
                                     * eltw_output_calibrations[f]
#    endif
        );

            output[out_idx] = TO_OUTPUT_TYPE_SAT(eltw_out);
#endif
            }// if(!zero_c)
        } // for (int c = 0; c < OUT_BLOCK_WIDTH; c++)
        }// if(!zero_r)
    } // for (int r = 0; r < OUT_BLOCK_HEIGHT; r++)
}

#if NON_BLOCK_LOAD != 1
#undef BLOCK_LOAD_WEIGHTS
#endif

#undef BLOCK_LOAD_INPUTS
#undef K_WSTRIDE
#undef K_HSTRIDE
#undef IN_BLOCK_WIDTH
#undef IN_BLOCK_HEIGHT
#undef PACK
#undef AS_TYPE_N_
#undef AS_TYPE_N
#undef AS_INPUT0_TYPE_4
#undef NUM_FILTERS
