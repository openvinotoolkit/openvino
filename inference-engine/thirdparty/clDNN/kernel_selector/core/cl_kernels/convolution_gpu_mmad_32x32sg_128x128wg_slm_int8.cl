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

#include "include/mmad.cl"

#define SCALE 0.11f

#ifdef LIGHTWEIGHT_QUANTIZATION

#define QUANTIZATION(idx) \
    {\
        for(uint z = 0; z < 4; z++)\
        {\
            regC_uchar16[z * 4 + 0] = convert_uchar_sat( (regC[0 * 4 + i][idx + z / 4]) * SCALE + bias_f.s0);\
            regC_uchar16[z * 4 + 1] = convert_uchar_sat( (regC[1 * 4 + i][idx + z / 4]) * SCALE + bias_f.s1);\
            regC_uchar16[z * 4 + 2] = convert_uchar_sat( (regC[2 * 4 + i][idx + z / 4]) * SCALE + bias_f.s2);\
            regC_uchar16[z * 4 + 3] = convert_uchar_sat( (regC[3 * 4 + i][idx + z / 4]) * SCALE + bias_f.s3);\
        }\
    }

#elif NO_QUANTIZATION

#define QUANTIZATION(idx) \
    regC_uchar16.s0 = convert_uchar_sat(regC[0 * 4 + i][idx]);\
    regC_uchar16.s1 = convert_uchar_sat(regC[1 * 4 + i][idx]);\
    regC_uchar16.s2 = convert_uchar_sat(regC[2 * 4 + i][idx]);\
    regC_uchar16.s3 = convert_uchar_sat(regC[3 * 4 + i][idx]);\
    \
    regC_uchar16.s4 = convert_uchar_sat(regC[0 * 4 + i][idx+1]);\
    regC_uchar16.s5 = convert_uchar_sat(regC[1 * 4 + i][idx+1]);\
    regC_uchar16.s6 = convert_uchar_sat(regC[2 * 4 + i][idx+1]);\
    regC_uchar16.s7 = convert_uchar_sat(regC[3 * 4 + i][idx+1]);\
    \
    regC_uchar16.s8 = convert_uchar_sat(regC[0 * 4 + i][idx+2]);\
    regC_uchar16.s9 = convert_uchar_sat(regC[1 * 4 + i][idx+2]);\
    regC_uchar16.sa = convert_uchar_sat(regC[2 * 4 + i][idx+2]);\
    regC_uchar16.sb = convert_uchar_sat(regC[3 * 4 + i][idx+2]);\
    \
    regC_uchar16.sc = convert_uchar_sat(regC[0 * 4 + i][idx+3]);\
    regC_uchar16.sd = convert_uchar_sat(regC[1 * 4 + i][idx+3]);\
    regC_uchar16.se = convert_uchar_sat(regC[2 * 4 + i][idx+3]);\
    regC_uchar16.sf = convert_uchar_sat(regC[3 * 4 + i][idx+3]);

#else

#define QUANTIZATION(idx) \
    regC_uchar16.s0 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i][idx]) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));\
    regC_uchar16.s1 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i][idx]) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));\
    regC_uchar16.s2 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i][idx]) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));\
    regC_uchar16.s3 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i][idx]) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));\
    \
    regC_uchar16.s4 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i][idx+1]) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));\
    regC_uchar16.s5 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i][idx+1]) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));\
    regC_uchar16.s6 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i][idx+1]) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));\
    regC_uchar16.s7 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i][idx+1]) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));\
    \
    regC_uchar16.s8 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i][idx+2]) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));\
    regC_uchar16.s9 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i][idx+2]) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));\
    regC_uchar16.sa = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i][idx+2]) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));\
    regC_uchar16.sb = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i][idx+2]) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));\
    \
    regC_uchar16.sc = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i][idx+3]) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));\
    regC_uchar16.sd = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i][idx+3]) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));\
    regC_uchar16.se = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i][idx+3]) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));\
    regC_uchar16.sf = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i][idx+3]) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));

#endif


inline uint FUNC(calculate_output_offset_to_account_padding)(uint cOffset)
{
#if OUT_WITH_PADDING == 1
    uint tmp_idx = cOffset;
    uint f_val_idx = tmp_idx % 32;
    tmp_idx /= 32;
    uint b_val_idx = tmp_idx % 4;
    tmp_idx /= 4;
    uint x_idx = tmp_idx % OUTPUT_SIZE_X;
    tmp_idx /= OUTPUT_SIZE_X;
    uint y_idx = tmp_idx % OUTPUT_SIZE_Y;
    tmp_idx /= OUTPUT_SIZE_Y;
    uint b_slice_idx = tmp_idx % (OUTPUT_BATCH_NUM / 4);
    tmp_idx /= (OUTPUT_BATCH_NUM / 4);
    uint f_slice_idx = tmp_idx % (OUTPUT_FEATURE_NUM / 32);

    uint padded_offset = f_slice_idx * OUT_F_BLOCK_PITCH;
    padded_offset += b_slice_idx * OUT_B_BLOCK_PITCH;
    padded_offset += y_idx * OUT_Y_PITCH;
    padded_offset += x_idx * OUT_X_PITCH;
    padded_offset += b_val_idx * 32;
    padded_offset += f_val_idx;
    padded_offset += OUT_OFFSET;

    return padded_offset;
#else
    return cOffset;
#endif
}

inline void FUNC(mmad_32x32_int8)(  __local uint* l_tileA, const uint l_offsetTileA,
                                    __local int8* l_tileB, const uint l_offsetTileB_col0,
                                    const uint l_offsetTileB_col1, const uint l_offsetTileB_col2,
                                    const uint l_offsetTileB_col3, int8* rowA, int8* colB,
                                    int8* regC)
{
    // Read tile A from SLM to regA
    uint l_offsetTileATemp = l_offsetTileA;
    __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
    for (uint j = 0; j < (SG_TILE_M / 8); ++j)
    {
        rowA[j] = as_int8(SLM_BLOCK_READ_8(&l_tileA[l_offsetTileATemp]));
        l_offsetTileATemp += 8 * SG_SIZE;
    }
    // Read tile B from SLM to regB and compute mmad
    colB[0] = l_tileB[l_offsetTileB_col0];
    colB[1] = l_tileB[l_offsetTileB_col1];
    __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
    for (uint j = 0; j < (SG_TILE_M / 8); ++j)
    {
        // Compute partial C
        regC[0*(SIMD_LANE_M / 8) + j] = MMAD_8x8( rowA[j], colB[0], regC[0*(SIMD_LANE_M / 8) + j]);
    }
    colB[0] = l_tileB[l_offsetTileB_col2];
    __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
    for (uint j = 0; j < (SG_TILE_M / 8); ++j)
    {
        // Compute partial C
        regC[1*(SIMD_LANE_M / 8) + j] = MMAD_8x8( rowA[j], colB[1], regC[1*(SIMD_LANE_M / 8) + j] );
    }
    colB[1] = l_tileB[l_offsetTileB_col3];
    __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
    for (uint j = 0; j < (SG_TILE_M / 8); ++j)
    {
        // Compute partial C
        regC[2*(SIMD_LANE_M / 8) + j] = MMAD_8x8(rowA[j], colB[0], regC[2*(SIMD_LANE_M / 8) + j]);
    }
    __attribute__((opencl_unroll_hint(SG_TILE_M / 8)))
    for (uint j = 0; j < (SG_TILE_M / 8); ++j)
    {
        // Compute partial C
        regC[3*(SIMD_LANE_M / 8) + j] = MMAD_8x8(rowA[j], colB[1], regC[3*(SIMD_LANE_M / 8) + j]);
    }
}

/*
 *  \brief GEMM kernel to compute MxN matrix using SLM
 *  \param g_inA  - Input matrix 
 *  \param g_inB  - Input matrix 
 *  \param g_outC - Output matrix
 */

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(Kernel_GEMM_MMAD8_32x32SG_128x128WG_SLM_INT8)
  (
  __global char* const g_inA,
  __global int* g_outC,
  __global char* const g_inB,
    #if BIAS_TERM
        __global BIAS_TYPE* biases,
    #endif
        __global float* quantizations,
    #if CALIBRATION_TERM
        __global float* calibrations,
    #endif
        uint split_idx

   )
{

    __global int4* const g_matrixA = (__global int4*)g_inA;
    __global int4* const g_matrixB = (__global int4*)g_inB;
    __global int8* g_matrixC = (__global int8*)g_outC;

    // Each work-group works to compute 128x128 tile.
    // Each work-group contains 16 sub-groups.
    // Each sub-group within the work-group works to compute a 32x32 tile.
    // 1) All work-items in WG fill SLM with tileA (128x32) and tileB (32x128).
    // 2) Each sub-group works to compute 32x32 tileC (stored in regC).
    //    Note that each work-item in the sub-group computes a 32x4 chunk of tileC.
    // 3) Repeat until tileC is fully computed (while moving tileA and tileB "windows")
    __local int8 l_workGroupTileA[2 * (WG_TILE_M * MATRIX_SMALL_K) / sizeof(int8)]; // [2*128*32/8] = 1024 
    __local int8 l_workGroupTileB[2 * (WG_TILE_N * MATRIX_SMALL_K) / sizeof(int8)]; // [2*128*32/8] = 1024 

    __local uint* l_workGroupTileA_uint = (__local uint*)l_workGroupTileA;
    __local int4* l_workGroupTileA_int4 = (__local int4*)l_workGroupTileA;
    __local int4* l_workGroupTileB_int4 = (__local int4*)l_workGroupTileB;

    const uint l_groupSize = (uint)get_local_size(DIM_X) * (uint)get_local_size(DIM_Y);

    const uint l_pingPongOffsetA_uint = (WG_TILE_M * MATRIX_SMALL_K) / sizeof(uint);
    const uint l_pingPongOffsetB_int8 = (WG_TILE_N * MATRIX_SMALL_K) / sizeof(int8);
    const uint l_pingPongOffsetA_int4 = (WG_TILE_M * MATRIX_SMALL_K) / sizeof(int4);
    const uint l_pingPongOffsetB_int4 = (WG_TILE_N * MATRIX_SMALL_K) / sizeof(int4);

    // Thread IDs
    const uint g_tidY = get_global_id(DIM_Y); // 0,...,all_wi_inY
    const uint g_tidX = get_global_id(DIM_X); // 0,...,all_wi_inX
    const uint l_tidX = get_local_id(DIM_X);  // 0,...,31 in WG
    const uint l_tidY = get_local_id(DIM_Y);  // 0,1,2,3  in WG
    const uint l_tid = l_tidY * (uint)get_local_size(DIM_X) + l_tidX; // 0,1,2,...127

    // SubGroup IDs
    const uint sg_tid = get_sub_group_local_id();            // 0,1,...,8
    const uint sg_global_idX = (uint)(g_tidX / SG_SIZE);     //{0}/8
    const uint sg_global_idY = g_tidY;                       //{0}

    const uint sg_local_idX = (uint)(l_tidX / SG_SIZE);      // {0,...,31}/8={0,0,0,0,0...,1,1,1,...,3,3,3}
    const uint sg_local_idY = l_tidY;                        // 0,1,2,3
    const uint sg_local_id = sg_local_idY * (uint)get_local_size(DIM_X) / SG_SIZE + sg_local_idX;  // get_local_size(DIM_X) / SG_SIZE = 32/8 = 4

    const uint sub_group_id = get_sub_group_id();


    // Registers
    int8 regC[(SIMD_LANE_M / 8) * SIMD_LANE_N] = {0}; // Each work-item responsible for 32x4 ints elts   // (32/8)*4
    int8 rowA[(SG_TILE_M * MATRIX_SMALL_K / SG_SIZE) / sizeof(int8)]; // each work-item will hold 1/8 of matrixA 
    int8 colB[2];  // each lane will store 32x4 piece of matrixB

    // SLM indices
    const uint l_offsetTileA = SG_TILE_M * (MATRIX_SMALL_K / sizeof(uint)) * sg_local_idY;
    const uint numElements32x32TileB = (MATRIX_SMALL_K * SG_TILE_N) / sizeof(int8);
    const uint numElements32x8TileB = numElements32x32TileB / 4;
    const uint l_offsetTileB = numElements32x32TileB * sg_local_idX;
    const uint l_offsetTileB_col0 = l_offsetTileB + sg_tid;
    const uint l_offsetTileB_col1 = l_offsetTileB + 1 * numElements32x8TileB + sg_tid;
    const uint l_offsetTileB_col2 = l_offsetTileB + 2 * numElements32x8TileB + sg_tid;
    const uint l_offsetTileB_col3 = l_offsetTileB + 3 * numElements32x8TileB + sg_tid;

    // Global indices
    uint g_idxA[2];
    uint g_idxB[2];
#ifdef TILED_GLOBAL_LAYOUT // 32-row major (matrixA) and 32-col major (matrixB)
    g_idxA[0] = ((MATRIX_SMALL_K / sizeof(int4)) * WG_TILE_M) * (uint)get_group_id(DIM_Y) + l_tid;
    g_idxB[0] = ((MATRIX_SMALL_K / sizeof(int4)) * WG_TILE_N) * (uint)get_group_id(DIM_X) + l_tid;
    g_idxA[1] = g_idxA[0] + l_groupSize;
    g_idxB[1] = g_idxB[0] + l_groupSize;
#else // Row (matrixA) and Col (matrixB) major layout
    g_idxA[0] = WG_TILE_M * (MATRIX_K / sizeof(int4)) * (uint)get_group_id(DIM_Y) +
               (l_tid / 2) * (MATRIX_K / sizeof(int4)) + (l_tid % 2);
    g_idxB[0] = WG_TILE_N * (MATRIX_K / sizeof(int4)) * (uint)get_group_id(DIM_X) +
               (l_tid / 2) * (MATRIX_K / sizeof(int4)) + (l_tid % 2);
    g_idxA[1] = g_idxA[0] + (l_groupSize / 2) * (MATRIX_K / sizeof(int4));
    g_idxB[1] = g_idxB[0] + (l_groupSize / 2) * (MATRIX_K / sizeof(int4));
#endif

    // Initial SLM setup
    {
        l_workGroupTileA_int4[l_tid] = g_matrixA[g_idxA[0]];
        l_workGroupTileB_int4[l_tid] = g_matrixB[g_idxB[0]];
        l_workGroupTileA_int4[l_tid + l_groupSize] = g_matrixA[g_idxA[1]];
        l_workGroupTileB_int4[l_tid + l_groupSize] = g_matrixB[g_idxB[1]];
	   
#ifdef TILED_GLOBAL_LAYOUT
        g_idxA[0] += MATRIX_M * MATRIX_SMALL_K / sizeof(int4);
        g_idxB[0] += MATRIX_N * MATRIX_SMALL_K / sizeof(int4);
        g_idxA[1] += MATRIX_M * MATRIX_SMALL_K / sizeof(int4);
        g_idxB[1] += MATRIX_N * MATRIX_SMALL_K / sizeof(int4);
#else
        g_idxA[0] += MATRIX_SMALL_K / sizeof(int4);
        g_idxB[0] += MATRIX_SMALL_K / sizeof(int4);
        g_idxA[1] += MATRIX_SMALL_K / sizeof(int4);
        g_idxB[1] += MATRIX_SMALL_K / sizeof(int4);
#endif

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int4 hdcReadValueA[2];
    int4 hdcReadValueB[2];

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < (MATRIX_K / MATRIX_SMALL_K) - 1; k++)
    {
        /*
         * SLM setup - HDC read only
         */
        // Overlap HDC reads with mmad compute
        hdcReadValueA[0] = g_matrixA[g_idxA[0]];
        hdcReadValueB[0] = g_matrixB[g_idxB[0]];
        hdcReadValueA[1] = g_matrixA[g_idxA[1]];
        hdcReadValueB[1] = g_matrixB[g_idxB[1]];

#ifdef TILED_GLOBAL_LAYOUT
        g_idxA[0] += MATRIX_M * MATRIX_SMALL_K / sizeof(int4);
        g_idxB[0] += MATRIX_N * MATRIX_SMALL_K / sizeof(int4);
        g_idxA[1] += MATRIX_M * MATRIX_SMALL_K / sizeof(int4);
        g_idxB[1] += MATRIX_N * MATRIX_SMALL_K / sizeof(int4);
#else
        g_idxA[0] += MATRIX_SMALL_K / sizeof(int4);
        g_idxB[0] += MATRIX_SMALL_K / sizeof(int4);
        g_idxA[1] += MATRIX_SMALL_K / sizeof(int4);
        g_idxB[1] += MATRIX_SMALL_K / sizeof(int4);
#endif

        /*
         * mmad compute
         */
        FUNC_CALL(mmad_32x32_int8)(&l_workGroupTileA_uint[(k % 2) * l_pingPongOffsetA_uint],
                                l_offsetTileA, &l_workGroupTileB[(k % 2) * l_pingPongOffsetB_int8],
                                l_offsetTileB_col0, l_offsetTileB_col1, l_offsetTileB_col2,
                                l_offsetTileB_col3, rowA, colB, regC);

        /*
         * SLM setup - SLM write only
         */
        l_workGroupTileA_int4[((k + 1) % 2 * l_pingPongOffsetA_int4) + l_tid] = hdcReadValueA[0];
        l_workGroupTileB_int4[((k + 1) % 2 * l_pingPongOffsetB_int4) + l_tid] = hdcReadValueB[0];
        l_workGroupTileA_int4[((k + 1) % 2 * l_pingPongOffsetA_int4) + l_tid + l_groupSize] = hdcReadValueA[1];
        l_workGroupTileB_int4[((k + 1) % 2 * l_pingPongOffsetB_int4) + l_tid + l_groupSize] = hdcReadValueB[1];

        barrier(CLK_LOCAL_MEM_FENCE);
    } // main outer loop

    /*
     * Last mmad compute iteration (avoids branching in main loop)
     */

    FUNC_CALL(mmad_32x32_int8)(
        &l_workGroupTileA_uint[(((MATRIX_K / MATRIX_SMALL_K) - 1) % 2) * l_pingPongOffsetA_uint],
        l_offsetTileA,
        &l_workGroupTileB[(((MATRIX_K / MATRIX_SMALL_K) - 1) % 2) * l_pingPongOffsetB_int8],
        l_offsetTileB_col0, l_offsetTileB_col1, l_offsetTileB_col2, l_offsetTileB_col3, rowA, colB,
        regC);

#ifdef OUTPUT_TILED_GLOBAL_LAYOUT
    // Write out in swizzled manner after quantizing
    __global uchar* g_outC_uchar = (__global uchar*)g_outC;
    uint cOffset = sg_global_idX * (MATRIX_M * SG_TILE_N / sizeof(uchar)) +
                   sg_global_idY * (SG_TILE_M * SG_TILE_N / sizeof(uchar));

    uchar16 regC_uchar16;
    uint offset_uc16 = 0;

    const uint workgroup_id_x = get_group_id(0); 
    uint feature_off = 32*(sub_group_id % (WG_TILE_N / 32)) + WG_TILE_N*workgroup_id_x; //=32*{0,1,2,3} + WG_TILE_N * workgroup_id_x 
    uint feature = get_sub_group_local_id()*4 + feature_off;

    float4 quant_f = vload4(0, quantizations + feature);
    float4 bias_f = vload4(0, biases + feature);
    float4 calib_f = vload4(0, calibrations + feature);

#if MMAD_SUPPORTED == 1
    __attribute__((opencl_unroll_hint( SG_TILE_M / (sizeof(int8) / sizeof(int)) )))
#endif
    for (uint i = 0; i < SG_TILE_M / (sizeof(int8) / sizeof(int)); i++)
    {
        uint padded_offset = FUNC_CALL(calculate_output_offset_to_account_padding)(cOffset);
        {
            // B0..3, F0..31
            QUANTIZATION(0);
        }

        intel_sub_group_block_write4((__global uint*)(g_outC_uchar + padded_offset), as_uint4(regC_uchar16));
        cOffset += sizeof(uchar16) * SG_SIZE;

        // now we need to calculate again for other x
        padded_offset = FUNC_CALL(calculate_output_offset_to_account_padding)(cOffset);
        {
            // B0..3, F0..31
            QUANTIZATION(4);
        }

        intel_sub_group_block_write4( (__global uint*)(g_outC_uchar + padded_offset), as_uint4(regC_uchar16) );
        cOffset += sizeof(uchar16) * SG_SIZE;
    }
#else
    // Write final accumulated values
    uint cOffset = sg_global_idX * ((MATRIX_M / 8) * SG_TILE_N) + sg_global_idY * (SG_TILE_M / 8) +
                   sg_tid * (MATRIX_M / 8);
    __attribute__((opencl_unroll_hint(SIMD_LANE_N)))
    for (uint i = 0; i < (SIMD_LANE_N); ++i)
    {
        __attribute__((opencl_unroll_hint(SIMD_LANE_M / 8)))
        for (uint j = 0; j < (SIMD_LANE_M / 8); ++j)
        {
            g_matrixC[cOffset + j] = regC[i*(SIMD_LANE_M / 8) + j];
        }
        cOffset += SG_SIZE * (MATRIX_M / 8);
    }
#endif

}

#undef QUANTIZATION
#undef SCALE
