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
#include "include/data_types.cl"
#include "include/imad.cl"

#ifndef NON_BLOCK_LOAD
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

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL (convolution_gpu_imad)(
    __global uint        *inputs,
    __global OUTPUT_TYPE *outputs,
    __global int         *weights
#if BIAS_TERM
    ,__global BIAS_TYPE  *biases
#endif
#if QUANTIZATION_TERM
    ,__global float      *quantizations
#endif
#if CALIBRATION_TERM
    ,__global float      *calibrations
#endif
)
{
    const uint oc = get_global_id(0) * OUT_BLOCK_WIDTH;  // oc = Output Column
    const uint or = get_global_id(1) * OUT_BLOCK_HEIGHT; // or = Output Row
    const uint fm = get_global_id(2);                    // fm = Feature Map = od = Output Depth, SIMD is across this dimension, WG is 1x1x16
    const uint fmg = get_group_id(2);
    const uint lid = get_local_id(2);
    const uint batch = fm / _OD;

    uint in[IN_BLOCK_HEIGHT];
    int  out[OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT] = { 0 };  // this is the 32 bit signed accumulator that must be converted to 8 bits before final write.

    #define NUM_FILTERS (K_HEIGHT * K_WIDTH)
    int w[NUM_FILTERS];

    int in_addr;

#ifdef BLOCK_LOAD_WEIGHTS
    int weight_addr = (fmg % (_OD / SIMD_SIZE)) * ((_ID * K_HEIGHT * K_WIDTH * SIMD_SIZE) / PACK);
#else
    int weight_addr = (fmg % (_OD / SIMD_SIZE)) * ((_ID * K_HEIGHT * K_WIDTH * SIMD_SIZE) / PACK) + lid;
#endif

    uint input_size = (_ID * (_IH + IHPAD) * (_IW + IWPAD)) / PACK; // dividing by PACK to get right number of 32bit entities.

    __attribute__((opencl_unroll_hint(1)))
    for(int kd = 0; kd < (_ID / PACK); kd++) // For imad we do 4X less input feature map iterations since we are packing 4 of them in each uchar4.  For now assume _ID is multiple of packing factor.
    {

#ifdef BLOCK_LOAD_INPUTS
        in_addr = kd * (_IH + IHPAD) * (_IW + IWPAD) + (or * K_STRIDE) * (_IW + IWPAD) + (oc * K_STRIDE);
#else
        in_addr = kd * (_IH + IHPAD) * (_IW + IWPAD) + (or * K_STRIDE) * (_IW + IWPAD) + (oc * K_STRIDE) + lid;
#endif
        in_addr += batch * input_size;  // adjust for batching

        for(uint reg = 0; reg < IN_BLOCK_HEIGHT; reg++) {
#ifdef BLOCK_LOAD_INPUTS
            in[reg] = intel_sub_group_block_read((const __global uint*) &inputs[in_addr]);
#else
            in[reg] = inputs[in_addr];// read SIMD_SIZE elements wide
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
        int kr = 0; // kr = Kernel Row
        LOOP(K_HEIGHT, kr,
        {
            int kc = 0; // kc = Kernel Column
            LOOP(K_WIDTH, kc,
            {
                for (int br = 0; br < OUT_BLOCK_HEIGHT; br++) {
                    for (int bc = 0; bc < OUT_BLOCK_WIDTH; bc++) {
                        uint input = sub_group_broadcast(in[br * K_HSTRIDE + kr], bc * K_WSTRIDE + kc);

                        out[br * OUT_BLOCK_WIDTH + bc] =
#ifdef CONVO_UNSIGNED
                            IMAD(out[br * OUT_BLOCK_WIDTH + bc], as_uchar4(input), as_char4(w[wi]));
#else
                            IMAD(out[br * OUT_BLOCK_WIDTH + bc], as_char4(input), as_char4(w[wi]));
#endif
                    }
                }
                wi++;
            });
        });
    } //for kd

    // Feature maps are an array of slices, each H,W position within the slice contains
    // four 8bit feature maps, packed like RGBA components into a 32 bit pixel.
    int row_size_bytes = (_OW + OWPAD) * PACK;

    // Slice_pack is a pack of 4 feature map tiles that are [OH][OW][4]
    // that are stored within the full [N][C/4][H][W][4] output.
    int slice_pack_size_bytes = row_size_bytes * (_OH + OHPAD);

    // Dividing the feature map index by 4 gives us the slice_pack_index in each lane
    // (each lane within block of 4 will have same index).
    int slice_pack_index = fm / PACK;

    // Each group of 4 simd lanes points to start of it's slice pack.
    int slice_pack_start_addr_bytes = slice_pack_index * slice_pack_size_bytes;

    // Make each lane within the group of 4(PACK) simd lanes point to an individual byte
    // witihn the uchar4 at start of slice pack.
    int slice_pack_addr_bytes = slice_pack_start_addr_bytes + (lid % PACK);

    // Adjust to particular tile that we are working on
    slice_pack_addr_bytes += (or + OUTPUT_PAD_BEFORE_SIZE_Y) * row_size_bytes
                             + (oc + OUTPUT_PAD_BEFORE_SIZE_X) * PACK;

    for (int r = 0; r < OUT_BLOCK_HEIGHT; r++) {
        for (int c = 0; c < OUT_BLOCK_WIDTH; c++) {
            uint out_idx = slice_pack_addr_bytes + r * row_size_bytes + (c*PACK);
#if QUANTIZATION_TERM
            int dotProd       = out[r * OUT_BLOCK_WIDTH + c];
#else
            UNIT_TYPE dotProd = out[r * OUT_BLOCK_WIDTH + c];
#endif

#if BIAS_TERM
            const uint f = fm % _OD;
    #if   BIAS_PER_OUTPUT
            #error convolution_gpu_imad.cl: BIAS_PER_OUTPUT - not supported
    #elif BIAS_PER_OFM
            const uint bias_index = f;
    #endif

    #if QUANTIZATION_TERM
        #if CALIBRATION_TERM

            dotProd = (UNIT_TYPE)round( ((float)dotProd * quantizations[f] * I_QF + biases[bias_index])
                                        * calibrations[f] );
        #else
            dotProd = (UNIT_TYPE)round( ((float)dotProd * quantizations[f] * I_QF + biases[bias_index])
                                        * O_QF );
        #endif // CALIBRATION_TERM
    #else
            dotProd += (UNIT_TYPE)biases[bias_index];
    #endif // QUANTIZATION_TERM
#endif // BIAS_TERM

#if QUANTIZATION_TERM
            UNIT_TYPE dotProd_A = ACTIVATION(convert_char(dotProd), NL_M, NL_N);
#else
            UNIT_TYPE dotProd_A = ACTIVATION(dotProd, NL_M, NL_N);
#endif

#ifdef CONVO_UNSIGNED
            outputs[out_idx] = (uchar)( max((int)dotProd_A , 0) & 0xFF );
#else
            outputs[out_idx] = (uchar)dotProd_A & 0xFF;
#endif
        } // for (int c = 0; c < OUT_BLOCK_WIDTH; c++)
    } // for (int r = 0; r < OUT_BLOCK_HEIGHT; r++)
}
