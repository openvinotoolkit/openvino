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

inline uint FUNC(calculate_output_offset_to_account_padding)(uint cOffset)
{
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

    // 1) All work-items in work-group fill SLM with tileA and tileB.
    // 2) Each sub-group works to compute a 32x32 tileC (stored in regC).
    //    Note that each work-item in the sub-group computes a 32x4 chunk of tileC.
    // 3) Repeat until tileC is fully computed (while moving tileA and tileB "windows")
    __local int8 l_workGroupTileA_0[(WG_TILE_M * MATRIX_SMALL_K) / sizeof(int8)];
    __local int8 l_workGroupTileB_0[(WG_TILE_N * MATRIX_SMALL_K) / sizeof(int8)];
    __local uint* l_workGroupTileA_uint_0 = (__local uint*)l_workGroupTileA_0;

    __local int8 l_workGroupTileA_1[(WG_TILE_M * MATRIX_SMALL_K) / sizeof(int8)];
    __local int8 l_workGroupTileB_1[(WG_TILE_N * MATRIX_SMALL_K) / sizeof(int8)];
    __local uint* l_workGroupTileA_uint_1 = (__local uint*)l_workGroupTileA_1;

    __local int8* l_workGroupTileA_live =  l_workGroupTileA_0;
    __local int8* l_workGroupTileB_live =  l_workGroupTileB_0;
    __local uint* l_workGroupTileA_live_uint = l_workGroupTileA_uint_0;

    __local int4* l_workGroupTileA_0_int4 = (__local int4*)l_workGroupTileA_0;
    __local int4* l_workGroupTileB_0_int4 = (__local int4*)l_workGroupTileB_0;
    __local int4* l_workGroupTileA_1_int4 = (__local int4*)l_workGroupTileA_1;
    __local int4* l_workGroupTileB_1_int4 = (__local int4*)l_workGroupTileB_1;

    const uint l_groupSize = (uint)get_local_size(DIM_X) * (uint)get_local_size(DIM_Y);

    // Thread IDs
    const uint g_tidY = get_global_id(DIM_Y);
    const uint g_tidX = get_global_id(DIM_X);
    const uint l_tidX = get_local_id(DIM_X);
    const uint l_tidY = get_local_id(DIM_Y);
    const uint l_tid = l_tidY * (uint)get_local_size(DIM_X) + l_tidX;

    // SubGroup IDs
    const uint sg_tid = get_sub_group_local_id();
    const uint sg_global_idX = (uint)(g_tidX / SG_SIZE);
    const uint sg_global_idY = g_tidY;
    const uint sg_local_idX = (uint)(l_tidX / SG_SIZE);
    const uint sg_local_idY = l_tidY;
    const uint sg_local_id = sg_local_idY * (uint)get_local_size(DIM_X) / SG_SIZE + sg_local_idX;

    const uint sub_group_id = get_sub_group_id();

    // Registers
    int8 regC[(SIMD_LANE_M / 8) * SIMD_LANE_N] = {0}; // Each work-item responsible for 32x4 ints elts
    int8 rowA[(SG_TILE_M * MATRIX_SMALL_K / SG_SIZE) / sizeof(int8)]; // each work-item will hold 1/8 of matrixA
    int8 colB[2]; // each lane will store 32x4 piece of matrixB

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
#ifdef TILED_GLOBAL_LAYOUT  // 32-row major (matrixA) and 32-col major (matrixB)
    uint g_idxA = ((MATRIX_SMALL_K / sizeof(int4)) * WG_TILE_M) * (uint)get_group_id(DIM_Y) + l_tid;
    uint g_idxB = ((MATRIX_SMALL_K / sizeof(int4)) * WG_TILE_N) * (uint)get_group_id(DIM_X) + l_tid;
#else  // Row (matrixA) and Col (matrixB) major layout
    uint g_idxA = WG_TILE_M * (MATRIX_K / sizeof(int4)) * (uint)get_group_id(DIM_Y) +
                  (l_tid / 2) * (MATRIX_K / sizeof(int4)) + (l_tid % 2);
    uint g_idxB = WG_TILE_N * (MATRIX_K / sizeof(int4)) * (uint)get_group_id(DIM_X) +
                  (l_tid / 2) * (MATRIX_K / sizeof(int4)) + (l_tid % 2);
#endif

    // Initial SLM setup
    {
        uint g_idxATemp = g_idxA;
        for (uint i = l_tid; i < (WG_TILE_M * MATRIX_SMALL_K / sizeof(int4)); i += WG_SIZE)
        {
            l_workGroupTileA_0_int4[i] = g_matrixA[g_idxATemp];
#ifdef TILED_GLOBAL_LAYOUT
            g_idxATemp += WG_SIZE;
#else
            g_idxATemp += (WG_SIZE / 2) * (MATRIX_K / sizeof(int4));
#endif
        }

        uint g_idxBTemp = g_idxB;
        for (uint i = l_tid; i < (WG_TILE_N * MATRIX_SMALL_K / sizeof(int4)); i += WG_SIZE)
        {
            l_workGroupTileB_0_int4[i] = g_matrixB[g_idxBTemp];
#ifdef TILED_GLOBAL_LAYOUT
            g_idxBTemp += WG_SIZE;
#else
            g_idxBTemp +=  (WG_SIZE / 2) * (MATRIX_K / sizeof(int4));
#endif
        }

#ifdef TILED_GLOBAL_LAYOUT
        g_idxA += MATRIX_M * MATRIX_SMALL_K / sizeof(int4);
        g_idxB += MATRIX_N * MATRIX_SMALL_K / sizeof(int4);
#else
        g_idxA += MATRIX_SMALL_K / sizeof(int4);
        g_idxB += MATRIX_SMALL_K / sizeof(int4);
#endif

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int4 hdcReadValueA[(WG_TILE_M * MATRIX_SMALL_K / sizeof(int4)) / WG_SIZE < 1
                           ? 1
                           : (WG_TILE_M * MATRIX_SMALL_K / sizeof(int4)) / WG_SIZE];
    int4 hdcReadValueB[(WG_TILE_N * MATRIX_SMALL_K / sizeof(int4)) / WG_SIZE < 1
                           ? 1
                           : (WG_TILE_N * MATRIX_SMALL_K / sizeof(int4)) / WG_SIZE];

    __attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < (MATRIX_K / MATRIX_SMALL_K) - 1; k++)
    {
        /*
         * SLM setup - HDC read only
         */

#if ((MATRIX_K / MATRIX_SMALL_K) > 1)
        uint g_idxATemp = g_idxA;
        for (uint i = l_tid, j = 0; i < (WG_TILE_M * MATRIX_SMALL_K / sizeof(int4)); i += WG_SIZE, ++j)
        {
            hdcReadValueA[j] = g_matrixA[g_idxATemp];
#ifdef TILED_GLOBAL_LAYOUT
            g_idxATemp += WG_SIZE;
#else
            g_idxATemp += (WG_SIZE / 2) * (MATRIX_K / sizeof(int4));
#endif
        }

        uint g_idxBTemp = g_idxB;
        for (uint i = l_tid, j = 0; i < (WG_TILE_N * MATRIX_SMALL_K / sizeof(int4)); i += WG_SIZE, ++j)
        {
            hdcReadValueB[j] = g_matrixB[g_idxBTemp];
#ifdef TILED_GLOBAL_LAYOUT
            g_idxBTemp += WG_SIZE;
#else
            g_idxBTemp += (WG_SIZE / 2) * (MATRIX_K / sizeof(int4));
#endif
        }

#ifdef TILED_GLOBAL_LAYOUT
        g_idxA += MATRIX_M * MATRIX_SMALL_K / sizeof(int4);
        g_idxB += MATRIX_N * MATRIX_SMALL_K / sizeof(int4);
#else
        g_idxA += MATRIX_SMALL_K / sizeof(int4);
        g_idxB += MATRIX_SMALL_K / sizeof(int4);
#endif
#endif

        /*
         * MMAD compute
         */

        FUNC_CALL(mmad_32x32_int8)(l_workGroupTileA_live_uint, l_offsetTileA, l_workGroupTileB_live,
                                l_offsetTileB_col0, l_offsetTileB_col1, l_offsetTileB_col2,
                                l_offsetTileB_col3, rowA, colB, regC);

        /*
         * SLM setup - SLM write only
         */

#if ((MATRIX_K / MATRIX_SMALL_K) > 1)
        if (k % 2 == 0)
        {
            for (uint i = l_tid, j = 0; i < (WG_TILE_M * MATRIX_SMALL_K / sizeof(int4));
                 i += WG_SIZE, ++j)
            {
                l_workGroupTileA_1_int4[i] = hdcReadValueA[j];
            }

            for (uint i = l_tid, j = 0; i < (WG_TILE_N * MATRIX_SMALL_K / sizeof(int4));
                 i += WG_SIZE, ++j)
            {
                l_workGroupTileB_1_int4[i] = hdcReadValueB[j];
            }

            l_workGroupTileA_live = l_workGroupTileA_1;
            l_workGroupTileB_live = l_workGroupTileB_1;
            l_workGroupTileA_live_uint = l_workGroupTileA_uint_1;
        }
        else
        {
            for (uint i = l_tid, j = 0; i < (WG_TILE_M * MATRIX_SMALL_K / sizeof(int4));
                 i += WG_SIZE, ++j)
            {
                l_workGroupTileA_0_int4[i] = hdcReadValueA[j];
            }

            for (uint i = l_tid, j = 0; i < (WG_TILE_N * MATRIX_SMALL_K / sizeof(int4));
                 i += WG_SIZE, ++j)
            {
                l_workGroupTileB_0_int4[i] = hdcReadValueB[j];
            }

            l_workGroupTileA_live = l_workGroupTileA_0;
            l_workGroupTileB_live = l_workGroupTileB_0;
            l_workGroupTileA_live_uint = l_workGroupTileA_uint_0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

    /*
     * Last MMAD compute iteration (avoids branching in main loop)
     */
    FUNC_CALL(mmad_32x32_int8)(l_workGroupTileA_live_uint, l_offsetTileA, l_workGroupTileB_live,
                            l_offsetTileB_col0, l_offsetTileB_col1, l_offsetTileB_col2,
                            l_offsetTileB_col3, rowA, colB, regC);
                            
#ifdef OUTPUT_TILED_GLOBAL_LAYOUT
    // Write out in swizzled manner after quantizing
    __global uchar* g_outC_uchar = (__global uchar*)g_outC;
    uint cOffset = sg_global_idX * (MATRIX_M * SG_TILE_N / sizeof(uchar)) +
                   sg_global_idY * (SG_TILE_M * SG_TILE_N / sizeof(uchar));

    uchar8 regC_uchar8[SIMD_LANE_M * SIMD_LANE_N / (sizeof(uchar8) / sizeof(uchar))];
    uint offset_uc8 = 0;

	const uint workgroup_id_x = get_group_id(0); 
	uint feature_off = 32*(sub_group_id % (WG_TILE_N / 32)) + WG_TILE_N*workgroup_id_x; //=32*{0,1,2,3} + WG_TILE_N * workgroup_id_x 
	uint feature = get_sub_group_local_id() + feature_off;

    float4 quant_f = as_float4(intel_sub_group_block_read4((__global uint*) (quantizations + feature) ));
    float4 bias_f = as_float4(intel_sub_group_block_read4((__global uint*) (biases + feature) ));
    float4 calib_f = as_float4(intel_sub_group_block_read4((__global uint*) (calibrations + feature) ));

#if MMAD_SUPPORTED == 1
    __attribute__((opencl_unroll_hint( SG_TILE_M / (sizeof(int8) / sizeof(int)) )))
#endif
    for (uint i = 0; i < SG_TILE_M / (sizeof(int8) / sizeof(int)); i++)
    {
        // begin of account for output PADDING
        uint padded_offset = FUNC_CALL(calculate_output_offset_to_account_padding)(cOffset);
        // end of account for padding

        // B0 F0..31
		regC_uchar8[offset_uc8].s0 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s0) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s1 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s0) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s2 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s0) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s3 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s0) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));
        // B1 F0..31		
		regC_uchar8[offset_uc8].s4 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s1) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s5 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s1) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s6 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s1) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s7 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s1) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));

		FUNC_CALL(sub_group_block_write_uchar8)(&g_outC_uchar[padded_offset], regC_uchar8[offset_uc8]);
        cOffset += sizeof(uchar8) * SG_SIZE;
        padded_offset += sizeof(uchar8) * SG_SIZE;
        offset_uc8++;

        // B2 F0..31
        regC_uchar8[offset_uc8].s0 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s2) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s1 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s2) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s2 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s2) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s3 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s2) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));
        // B3 F0..31		
		regC_uchar8[offset_uc8].s4 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s3) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s5 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s3) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s6 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s3) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s7 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s3) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));
		
		FUNC_CALL(sub_group_block_write_uchar8)(&g_outC_uchar[padded_offset], regC_uchar8[offset_uc8]);
		cOffset += sizeof(uchar8) * SG_SIZE;
        offset_uc8++;

        // now we need to calculate again for other x
        padded_offset = FUNC_CALL(calculate_output_offset_to_account_padding)(cOffset);
        //

        regC_uchar8[offset_uc8].s0 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s4) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s1 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s4) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s2 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s4) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s3 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s4) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));
		
		regC_uchar8[offset_uc8].s4 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s5) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s5 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s5) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s6 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s5) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s7 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s5) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));

		FUNC_CALL(sub_group_block_write_uchar8)(&g_outC_uchar[padded_offset], regC_uchar8[offset_uc8]);
        cOffset += sizeof(uchar8) * SG_SIZE;
        padded_offset += sizeof(uchar8) * SG_SIZE;
        offset_uc8++;

        regC_uchar8[offset_uc8].s0 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s6) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s1 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s6) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s2 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s6) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s3 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s6) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));
		
		regC_uchar8[offset_uc8].s4 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[0 * 4 + i].s7) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s5 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[1 * 4 + i].s7) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s6 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[2 * 4 + i].s7) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), ACTIVATION_PARAMS));
		regC_uchar8[offset_uc8].s7 = as_uchar(ACTIVATION( convert_char(round(( (float)(regC[3 * 4 + i].s7) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), ACTIVATION_PARAMS));

		FUNC_CALL(sub_group_block_write_uchar8)(&g_outC_uchar[padded_offset], regC_uchar8[offset_uc8]);
        cOffset += sizeof(uchar8) * SG_SIZE;
        offset_uc8++;
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
