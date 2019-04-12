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

#include "include/mmad.cl"

#define SCALE 0.11f

#ifdef LIGHTWEIGHT_QUANTIZATION

#define QUANTIZATION \
    out_write_N2K4[0].s0 = convert_uchar_sat((float)outvec0.s0 * SCALE + bias_f.s0); /*K= lane_id,N=0*/ \
    out_write_N2K4[0].s1 = convert_uchar_sat((float)outvec1.s0 * SCALE + bias_f.s1); /*K= lane_id + 8,N=0*/\
    out_write_N2K4[0].s2 = convert_uchar_sat((float)outvec2.s0 * SCALE + bias_f.s2); /*K= lane_id + 16,N=0*/\
    out_write_N2K4[0].s3 = convert_uchar_sat((float)outvec3.s0 * SCALE + bias_f.s3); /*K= lane_id + 24,N=0*/\
    \    
    out_write_N2K4[0].s4 = convert_uchar_sat((float)outvec0.s1 * SCALE + bias_f.s0); /*K= lane_id,N=1*/\
    out_write_N2K4[0].s5 = convert_uchar_sat((float)outvec1.s1 * SCALE + bias_f.s1); /*K= lane_id + 8,N=1*/\
    out_write_N2K4[0].s6 = convert_uchar_sat((float)outvec2.s1 * SCALE + bias_f.s2); /*K= lane_id + 16,N=1*/\
    out_write_N2K4[0].s7 = convert_uchar_sat((float)outvec3.s1 * SCALE + bias_f.s3); /*K= lane_id + 24,N=1*/\
    \
    out_write_N2K4[1].s0 = convert_uchar_sat((float)outvec0.s2 * SCALE + bias_f.s0); /*K= lane_id,N=2*/\
    out_write_N2K4[1].s1 = convert_uchar_sat((float)outvec1.s2 * SCALE + bias_f.s1); /*K= lane_id + 8,N=2*/\
    out_write_N2K4[1].s2 = convert_uchar_sat((float)outvec2.s2 * SCALE + bias_f.s2); /*K= lane_id + 16,N=2*/\
    out_write_N2K4[1].s3 = convert_uchar_sat((float)outvec3.s2 * SCALE + bias_f.s3); /*K= lane_id + 24,N=2*/\
    \
    out_write_N2K4[1].s4 = convert_uchar_sat((float)outvec0.s3 * SCALE + bias_f.s0); /*K= lane_id,N=3*/\
    out_write_N2K4[1].s5 = convert_uchar_sat((float)outvec1.s3 * SCALE + bias_f.s1); /*K= lane_id + 8,N=3*/\
    out_write_N2K4[1].s6 = convert_uchar_sat((float)outvec2.s3 * SCALE + bias_f.s2); /*K= lane_id + 16,N=3*/\
    out_write_N2K4[1].s7 = convert_uchar_sat((float)outvec3.s3 * SCALE + bias_f.s3); /*K= lane_id + 24,N=3*/

#elif NO_QUANTIZATION

#define QUANTIZATION \
    out_write_N2K4[0].s0 = convert_uchar_sat(outvec0.s0); /*K= lane_id,N=0*/ \
    out_write_N2K4[0].s1 = convert_uchar_sat(outvec1.s0); /*K= lane_id + 8,N=0*/\
    out_write_N2K4[0].s2 = convert_uchar_sat(outvec2.s0); /*K= lane_id + 16,N=0*/\
    out_write_N2K4[0].s3 = convert_uchar_sat(outvec3.s0); /*K= lane_id + 24,N=0*/\
    \    
    out_write_N2K4[0].s4 = convert_uchar_sat(outvec0.s1); /*K= lane_id,N=1*/\
    out_write_N2K4[0].s5 = convert_uchar_sat(outvec1.s1); /*K= lane_id + 8,N=1*/\
    out_write_N2K4[0].s6 = convert_uchar_sat(outvec2.s1); /*K= lane_id + 16,N=1*/\
    out_write_N2K4[0].s7 = convert_uchar_sat(outvec3.s1); /*K= lane_id + 24,N=1*/\
    \
    out_write_N2K4[1].s0 = convert_uchar_sat(outvec0.s2); /*K= lane_id,N=2*/\
    out_write_N2K4[1].s1 = convert_uchar_sat(outvec1.s2); /*K= lane_id + 8,N=2*/\
    out_write_N2K4[1].s2 = convert_uchar_sat(outvec2.s2); /*K= lane_id + 16,N=2*/\
    out_write_N2K4[1].s3 = convert_uchar_sat(outvec3.s2); /*K= lane_id + 24,N=2*/\
    \
    out_write_N2K4[1].s4 = convert_uchar_sat(outvec0.s3); /*K= lane_id,N=3*/\
    out_write_N2K4[1].s5 = convert_uchar_sat(outvec1.s3); /*K= lane_id + 8,N=3*/\
    out_write_N2K4[1].s6 = convert_uchar_sat(outvec2.s3); /*K= lane_id + 16,N=3*/\
    out_write_N2K4[1].s7 = convert_uchar_sat(outvec3.s3); /*K= lane_id + 24,N=3*/

#else

#define QUANTIZATION \
    out_write_N2K4[0].s0 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec0.s0) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), NL_M, NL_N)); /*K= lane_id,N=0*/ \
    out_write_N2K4[0].s1 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec1.s0) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), NL_M, NL_N)); /*K= lane_id + 8,N=0*/\
    out_write_N2K4[0].s2 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec2.s0) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), NL_M, NL_N)); /*K= lane_id + 16,N=0*/\
    out_write_N2K4[0].s3 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec3.s0) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), NL_M, NL_N)); /*K= lane_id + 24,N=0*/\
    \    
    out_write_N2K4[0].s4 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec0.s1) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), NL_M, NL_N)); /*K= lane_id,N=1*/\
    out_write_N2K4[0].s5 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec1.s1) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), NL_M, NL_N)); /*K= lane_id + 8,N=1*/\
    out_write_N2K4[0].s6 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec2.s1) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), NL_M, NL_N)); /*K= lane_id + 16,N=1*/\
    out_write_N2K4[0].s7 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec3.s1) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), NL_M, NL_N)); /*K= lane_id + 24,N=1*/\
    \
    out_write_N2K4[1].s0 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec0.s2) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), NL_M, NL_N)); /*K= lane_id,N=2*/\
    out_write_N2K4[1].s1 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec1.s2) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), NL_M, NL_N)); /*K= lane_id + 8,N=2*/\
    out_write_N2K4[1].s2 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec2.s2) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), NL_M, NL_N)); /*K= lane_id + 16,N=2*/\
    out_write_N2K4[1].s3 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec3.s2) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), NL_M, NL_N)); /*K= lane_id + 24,N=2*/\
    \
    out_write_N2K4[1].s4 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec0.s3) * quant_f.s0 * I_QF + bias_f.s0) * calib_f.s0)), NL_M, NL_N)); /*K= lane_id,N=3*/\
    out_write_N2K4[1].s5 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec1.s3) * quant_f.s1 * I_QF + bias_f.s1) * calib_f.s1)), NL_M, NL_N)); /*K= lane_id + 8,N=3*/\
    out_write_N2K4[1].s6 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec2.s3) * quant_f.s2 * I_QF + bias_f.s2) * calib_f.s2)), NL_M, NL_N)); /*K= lane_id + 16,N=3*/\
    out_write_N2K4[1].s7 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec3.s3) * quant_f.s3 * I_QF + bias_f.s3) * calib_f.s3)), NL_M, NL_N)); /*K= lane_id + 24,N=3*/

#endif

// mapping to clDNN
#define _MMAD_4x8(C, A, B) MMAD_4x8(A, B, C)
#define _OD OUTPUT_FEATURE_NUM
#define _OW OUTPUT_SIZE_X
#define _OH OUTPUT_SIZE_Y
#define OWPAD (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X)
#define OHPAD (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y)
#define _IH INPUT0_SIZE_Y
#define _IW INPUT0_SIZE_X
#define _ID INPUT0_FEATURE_NUM
#define K_HEIGHT FILTER_SIZE_Y
#define K_WIDTH FILTER_SIZE_X
#define BATCH_SIZE OUTPUT_BATCH_NUM

#define IHPAD (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y)
#define IWPAD (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_PAD_AFTER_SIZE_X)
#define K_STRIDE STRIDE_SIZE_X
// end of mapping

// for now kernel stride is square
#define K_WSTRIDE K_STRIDE
#define K_HSTRIDE K_STRIDE

#define PACK 32
#define BATCH_PACK 4

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(convolution_mmad_slm_2x14_rep4)(
__global int8 *inputs,
__global uchar* outputs,
__global int8* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if QUANTIZATION_TERM
    const __global float* quantizations,
#endif
#if CALIBRATION_TERM
    const __global float* calibrations,
#endif
    uint split_idx
)
{
	const uint TILE_H = OUT_BLOCK_HEIGHT*LOCAL_SIZE_Z;
	const uint TILE_W = OUT_BLOCK_WIDTH*LOCAL_SIZE_Y;

	ushort fmg     = get_group_id(0);   // Output Depth
	ushort group_y = get_group_id(1);   // Output Width
	ushort group_z = get_group_id(2);   // Output Height

	/* 16,1,8 WG , SIMD8 - 16 HW threads in a WG
	threads 0-1 : ( lid_x:0-15,lid_y:0,lid_z:0)
	threads 2-3 : ( lid_x:0-15,lid_y:0,lid_z:1)
	..
	threads 12-13: ( lid_x:0-15, lid_y:0,lid_z:6)
	threads 14-15: ( lid_x:0-15, lid_y:0,lid_z:7)
	*/

	/* Thread, local IDs */
	ushort thread_id 		= get_sub_group_id();
	ushort threadid_mod_2   = thread_id % 2;
	ushort threadid_mod_8   = thread_id % 8;

	ushort lid_x    = get_local_id(0);
	ushort lid_z    = get_local_id(2);

	uchar  lane_id  = get_sub_group_local_id();

	/* 32-bit signed accumulator , 112 output registers for 1Px7Qx4Nx32K output tile size
	   Will be converted to 8-bits before final write */

	int4  out_07 [ OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH ]   = {0}; // For output channels 0-7
	int4  out_815[ OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH ]   = {0}; // For output channels 8-15
	int4  out_1623[ OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH ]  = {0}; // For output channels 16-23
	int4  out_2431[ OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH ]  = {0}; // For output channels 24-31

	/* Account for batching */

	ushort batch 	= ( fmg*LOCAL_SIZE_X*4 ) /_OD; // Each thread processing 32 output_channels and each fmg processing 64 output channels , LOCAL_SIZE_X is only 16

	// Size calculated for int8 elements
	uint input_size = (_IH + IHPAD) * (_IW + IWPAD) * BATCH_PACK ;

	uint in_addr_offset = batch*input_size;

	/* Goto activation tile for work group, offset is w.r.t int8 array */

	uint groupy_tile = TILE_W*group_y;
	uint groupz_tile = TILE_H*group_z;

    in_addr_offset += (groupz_tile * K_STRIDE) * (_IW + IWPAD) * BATCH_PACK + (groupy_tile * K_STRIDE) * BATCH_PACK;

	 	/* SLM space for Activation, Weights
	       ( 16,1,8 ) Workgroup - 7 tiles along Y direction and 64 different output channels
		    2 threads used to load global memory
	        Activation - 9Hx9Wx4Nx32C Weights -3Rx3Sx64Kx32C	*/

	__local int8 act_slm      [  9*9*4 ];
	__local int8 weight_slm   [  9*64  ];

   /* 9Hx9Wx4Nx32C activation tile written into SLM.  Distribute among 14 threads in Workgroup
	   threads 0-1 write 9x4x32 of  H=0, W=0...8
	   threads 2-3 write 9x4x32 of H=1, W=0...8
	   threads 4-5 write 9x4x32 of H=2, W=0...8
	   threads 6-7  write 9x4x32 of H=3, W=0...8
	   threads 8-9 write 9x4x32 of H=4, W=0...8
	   threads 10-11 write 9x4x32 of H=5,W=0...8
	   threads 12-13 write 9x4x32 of H=6,W=0...8
	   threads 14 write 9x4x32 of H=7,W=0...8
	   threads 15 write 9x4x32 of H=8,W=0...8 */

	/* Goto activation tile for thread in group */

	uint row_offset   =  thread_id / 2;

	if ( thread_id >= 14 )
    {
        row_offset = 7;
	}

	// In addr offset for the particular thread
	in_addr_offset    += row_offset * K_STRIDE * (_IW + IWPAD ) * BATCH_PACK ;

   /* Activation SLM indices */
    uint act_slm_write =  row_offset * ( TILE_W + 2) * BATCH_PACK;
	uint act_slm_read  =  OUT_BLOCK_HEIGHT * lid_z * ( TILE_W + 2) * BATCH_PACK ;

	/* 9RSx64Kx32C Weight Block in SLM
	   thread0 handles ( reads from global ) w(0,0),w(0,1),w(0,2) of K=0,1 ( k=0..15 )
	   thread1 handles w(0,0),w(0,1),w(0,2) of K=2,3 ( k=16..31)
	   thread2 handles w(1,0),w(1,1) of K=0,1 ( k=0..15)
	   thread3 handles w(1,0),w(1,1) of K=2,3 ( k=16..31)
	   thread4 handles w(1,2),w(2,0) of K=0,1 ( k=0..15)
	   thread5 handles w(1,2),w(2,0) of K=2,3 ( k=16..31)
	   thread6 handles w(2,1),w(2,2) of K=0,1 ( k=0..15)
	   thread7 handles w(2,1),w(2,2) of K=2,3 ( k=16..31)

	   Similarly threads8-15 handles for K=4,5,6,7

	   Weight Layout in SLM

	   w(R=0,S=0,k=0..7,C=0..15),w(R=0,S=0,k=32..39,C=0..15)
	   w(R=0,S=0,k=0..7,C=16..31),w(R=0,S=0,k=32..39,C=16..31)

	   Above interleaving present to avoid SLM Bank conflicts when fused threads read from SLM
	   Thread0 will read k=0..31, thread1 will read k=32..63

	   First all output channels are present in SLM, then next weight pixel is present in SLM */

	 #define NUM_FILTERS (K_HEIGHT * K_WIDTH)

	 uint output_depth    = fmg % ( _OD / ( LOCAL_SIZE_X * 4 ) ); //LOCAL_SIZE_X=16, 64 output channels used

	 uint weight_size_CRS =  ( _ID / PACK ) * NUM_FILTERS * 8; //8 output channels packed inside

	 // Global weight addr for workgroup
	 uint weight_global_addr_offset =  output_depth * 8 * weight_size_CRS ; //64 output channels per workgroup

	 /* Global weight address for thread */

	 // Goto appropriate output channel in weights
	 uint weight_global_channel_offset = threadid_mod_2 * 2 * weight_size_CRS ;

	uint slm_channel_offset     = threadid_mod_2;
	uint bc_fused_thread_offset = 0;

	 if ( thread_id >= 8 )
    {
		bc_fused_thread_offset =  1;

		weight_global_channel_offset =  4 * weight_size_CRS + slm_channel_offset * weight_size_CRS * 2 ;
    }

	 // Goto appropriate pixel in weights

	 uint weight_global_pixel_offset = 0;
	 uint slm_pixel_offset = 0;

    if ( threadid_mod_8 >=2  )
    {
	 /* First three pixels handled by threads 0-1, then 2 pixels handled by two threads */

		weight_global_pixel_offset = 3*8 +  ( ( (threadid_mod_8/2) - 1 )*2*8 );
		slm_pixel_offset 		   = 3*64 + ( ( (threadid_mod_8/2) - 1 )*2*64 );
    }

    weight_global_addr_offset += weight_global_channel_offset + weight_global_pixel_offset;

	 /* Weight slm write index */

	 uint slm_write_weight = slm_pixel_offset + slm_channel_offset * 32 + bc_fused_thread_offset * 4;

	 /* Weight slm read index */

	 /* Thread 0  reads output channels 0-15, thread 1 handles output channels 16-31, data present in interleaved
	    manner in SLM
		Data layout in SLM

		w(0,0) C=0..7, K = 0..7 | w(0,0) C=0..7, K = 32..39
		w(0,0) C=8..15,K=0..7   | w(0,0) C=8..15,K = 32..39
		w(0,0) C=0..7, K=8..15  | w(0,0) C=0..7, K = 40..47
		w(0,0) C=8..15,K=8..15  | w(0,0) C=8..15,K=  40..47

		*/
    uint wt_slm_rd_offset = threadid_mod_2*4;

	int kd;

	__attribute__((opencl_unroll_hint(1)))
	for(kd = 0; kd <  ( _ID / PACK ) ; kd++)
	{
		{
			/* Load Activation from global to SLM */

			int in_addr = kd * (_IH + IHPAD) * (_IW + IWPAD) * BATCH_SIZE + in_addr_offset;

			__global uint *activation_tile = (__global uint*)&inputs[ in_addr ];

			__local uint *act_slm_ptr   = (__local uint *) &act_slm [ act_slm_write  ];

			/* The odd thread in fused pair will start from next 4x8 block */

			activation_tile += threadid_mod_2*4*8;
			act_slm_ptr 	+= threadid_mod_2*4*8;

			int4 act_col_0 =  as_int4( intel_sub_group_block_read4(activation_tile) );//col 0
			int4 act_col_1 =  as_int4( intel_sub_group_block_read4(activation_tile + 8*8) );//col 2
			int4 act_col_2 =  as_int4( intel_sub_group_block_read4(activation_tile + 2*8*8) );//col 4
			int4 act_col_3 =  as_int4( intel_sub_group_block_read4(activation_tile + 3*8*8) );//col 6

			SLM_BLOCK_WRITE_4 ( act_slm_ptr , as_uint4 ( act_col_0 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 8*8 ) , as_uint4 ( act_col_1 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 2*8*8 ) , as_uint4 ( act_col_2 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 3*8*8 ) , as_uint4 ( act_col_3 ) );

			if ( threadid_mod_2  == 0 )
            {
				int4 act_col_4 =  as_int4( intel_sub_group_block_read4(activation_tile + 4*8*8) );

				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 4*8*8 ) , as_uint4 ( act_col_4 ) );
			}

			if ( thread_id >=14)
            {
				activation_tile  = activation_tile + 1 * (_IW + IWPAD ) * BATCH_PACK * 8;
				act_slm_ptr 	 = act_slm_ptr + (TILE_W + 2)  * BATCH_PACK *8;

				int4 act_col_9 =  as_int4( intel_sub_group_block_read4(activation_tile) );
				int4 act_col_10 =  as_int4( intel_sub_group_block_read4(activation_tile + 8*8) );
				int4 act_col_11 =  as_int4( intel_sub_group_block_read4(activation_tile + 2*8*8) );
				int4 act_col_12 =  as_int4( intel_sub_group_block_read4(activation_tile + 3*8*8) );

				SLM_BLOCK_WRITE_4 ( act_slm_ptr  , as_uint4 ( act_col_9 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 8*8 )   , as_uint4 ( act_col_10 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 2*8*8 ) , as_uint4 ( act_col_11 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 3*8*8 ) , as_uint4 ( act_col_12 ) );

				if ( threadid_mod_2  == 0 )
                {
					int4 act_col_13 =  as_int4( intel_sub_group_block_read4(activation_tile + 4*8*8) );

					SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 4*8*8 ) , as_uint4 ( act_col_13 ) );
				}
			}

		/* load weights from global to weight_slm */

			int weight_addr = kd * NUM_FILTERS * 8 + weight_global_addr_offset;

			__global uint *weight_tile   = (__global uint*)&weights    [ weight_addr ];
			__local  uint *wt_slm_ptr    = (__local uint *)&weight_slm [ slm_write_weight  ];

			__global uint *weight_tile_2   = weight_tile;
			__local uint *wt_slm_ptr_2     = wt_slm_ptr;

			int4 w0 = as_int4 ( intel_sub_group_block_read4( weight_tile ) );	// Pixel1 K=0..7 C=0..15
			int4 w1 = as_int4 ( intel_sub_group_block_read4( weight_tile + 4*8 ) );	// Pixel1 K=0..7 C=16..31
			int4 w2 = as_int4 ( intel_sub_group_block_read4( weight_tile + 8*8 ) );	// Pixel2 K=0..7 C=0..15
			int4 w3 = as_int4 ( intel_sub_group_block_read4( weight_tile + 12*8 ) );// Pixel2 K=0..7 C=16..31

			// Goto next output channel
			weight_tile += weight_size_CRS*8;

			int4 w4 = as_int4 ( intel_sub_group_block_read4( weight_tile ) );	// Pixel1 K=8..15 C=0..15
			int4 w5 = as_int4 ( intel_sub_group_block_read4( weight_tile + 4*8 ) );	// Pixel1 K=8..15 C=16..31
			int4 w6 = as_int4 ( intel_sub_group_block_read4( weight_tile + 8*8 ) );	// Pixel2 K=8..15 C=0..15
			int4 w7 = as_int4 ( intel_sub_group_block_read4( weight_tile + 12*8 ) );// Pixel2 K=8..15 C=16..31

			SLM_BLOCK_WRITE_4 ( wt_slm_ptr, as_uint4 ( w0 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 8*8 ) , as_uint4 ( w1 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 64*8 ), as_uint4 ( w2 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 64*8 + 8*8 ), as_uint4 ( w3 ) );

			wt_slm_ptr  += 16*8;

			SLM_BLOCK_WRITE_4 ( wt_slm_ptr , as_uint4 ( w4 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 8*8 )   , as_uint4 ( w5 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 64*8 ) , as_uint4 ( w6 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 64*8 + 8*8 ) , as_uint4 ( w7 ) );

		   if( threadid_mod_8 < 2 )
           {
				// Goto next pixel
				weight_tile_2 += 16*8;
				wt_slm_ptr_2  += 2*64*8;

				int4 w0 = as_int4 ( intel_sub_group_block_read4( weight_tile_2 ) );	// Pixel1 K=0..7 C=0..15
				int4 w1 = as_int4 ( intel_sub_group_block_read4( weight_tile_2 + 4*8 ) );	// Pixel1 K=0..7 C=16..31

				// Goto next output channel
				weight_tile_2 += weight_size_CRS*8;

				int4 w4 = as_int4 ( intel_sub_group_block_read4( weight_tile_2 ) );	// Pixel1 K=8..15 C=0..15
				int4 w5 = as_int4 ( intel_sub_group_block_read4( weight_tile_2 + 4*8 ) );	// Pixel1 K=8..15 C=16..31

				SLM_BLOCK_WRITE_4 ( wt_slm_ptr_2, as_uint4 ( w0 ) );
				SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr_2 + 8*8 ) , as_uint4 ( w1 ) );

				wt_slm_ptr_2  += 16*8;

				SLM_BLOCK_WRITE_4 ( wt_slm_ptr_2 , as_uint4 ( w4 ) );
				SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr_2 + 8*8 )   , as_uint4 ( w5 ) );
			}
	}

		// Synchronize SLM writes across workgroup
		 barrier(CLK_LOCAL_MEM_FENCE);

		if ( lid_z <= 6 )
        {
			uint wt_slm_rd = wt_slm_rd_offset;

			__local uint *slm_ptr0     = (__local uint *) &act_slm[ act_slm_read ];
			__local uint *slm_ptr1     = (__local uint *) &weight_slm[ wt_slm_rd ];

			/* balancing load of weights, activations   */
			int8 weights_reg[3]; //24 registers
			int4 act_reg[18];    //72 registers
			uint slm_read_pixel_offset = 64*8;

			/**********************************************************************************************************
			  First phase - multiply first row of weights  and 1st row of activations
			***********************************************************************************************************/

	                 /* Load weights from SLM into registers - row0, output channels 0..7  */

				{
					 	__local uint *slm_ptrw0  = slm_ptr1;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
				}

			/* load 1Hx9Wx4N inputs, Activation row0   */

				__attribute__((opencl_unroll_hint(9)))
				for (int ic = 0; ic < 9; ic++)
				{
	                 /* Load activations from SLM into registers  */

					 uint slm_offset = ic * BATCH_PACK * 8 ;

    				 act_reg [ ic ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ;
				}

			/* Convolve */

			   /* order the mmad instructions to minimize dependency on src0,dst - also try to maximise reuse of weights-reg*/

				/*  Output channels 0-7 */

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[0], weights_reg[0] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[1], weights_reg[0] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[2], weights_reg[0] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[3], weights_reg[0] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[4], weights_reg[0] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[5], weights_reg[0] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[6], weights_reg[0] );

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[1], weights_reg[1] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[2], weights_reg[1] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[3], weights_reg[1] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[4], weights_reg[1] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[5], weights_reg[1] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[6], weights_reg[1] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[7], weights_reg[1] );

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[2], weights_reg[2] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[3], weights_reg[2] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[4], weights_reg[2] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[5], weights_reg[2] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[6], weights_reg[2] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[7], weights_reg[2] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[8], weights_reg[2] );

		     /* Load weights from SLM into registers - row0, output channels 8..15  */

				{
					 	__local uint *slm_ptrw0 = slm_ptr1 + 2*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
				}

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[0], weights_reg[0] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[1], weights_reg[0] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[2], weights_reg[0] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[3], weights_reg[0] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[4], weights_reg[0] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[5], weights_reg[0] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[6], weights_reg[0] );

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[1], weights_reg[1] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[2], weights_reg[1] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[3], weights_reg[1] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[4], weights_reg[1] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[5], weights_reg[1] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[6], weights_reg[1] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[7], weights_reg[1] );

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[2], weights_reg[2] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[3], weights_reg[2] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[4], weights_reg[2] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[5], weights_reg[2] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[6], weights_reg[2] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[7], weights_reg[2] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[8], weights_reg[2] );

				/* Load weights from SLM into registers - row0, output channels 16..23  */
				{
					 	__local uint *slm_ptrw0 = slm_ptr1 + 4*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
				}

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[0], weights_reg[0] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[1], weights_reg[0] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[2], weights_reg[0] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[3], weights_reg[0] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[4], weights_reg[0] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[5], weights_reg[0] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[6], weights_reg[0] );

				/* load 1Hx9Wx4N inputs, Activation row1   */

				uint slm_row_offset_2 	  = 1*(TILE_W + 2)*BATCH_PACK*8;

				__attribute__((opencl_unroll_hint(9)))
				for (int ic = 0; ic < 9; ic++)
				{
	                 /* Load activations from SLM into registers  */

					 uint slm_offset = slm_row_offset_2 + ic * BATCH_PACK * 8 ;

    				 act_reg [ ic + 9 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ;
				}

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[1], weights_reg[1] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[2], weights_reg[1] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[3], weights_reg[1] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[4], weights_reg[1] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[5], weights_reg[1] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[6], weights_reg[1] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[7], weights_reg[1] );

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[2], weights_reg[2] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[3], weights_reg[2] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[4], weights_reg[2] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[5], weights_reg[2] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[6], weights_reg[2] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[7], weights_reg[2] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[8], weights_reg[2] );

				/* Load weights from SLM into registers - row0, output channels 24..31  */
				{
					 	__local uint *slm_ptrw0 = slm_ptr1 + 6*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
						slm_ptrw0   			 += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw0 + 64 ) );
				}

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[0], weights_reg[0] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[1], weights_reg[0] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[2], weights_reg[0] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[3], weights_reg[0] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[4], weights_reg[0] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[5], weights_reg[0] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[6], weights_reg[0] );

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[1], weights_reg[1] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[2], weights_reg[1] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[3], weights_reg[1] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[4], weights_reg[1] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[5], weights_reg[1] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[6], weights_reg[1] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[7], weights_reg[1] );

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[2], weights_reg[2] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[3], weights_reg[2] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[4], weights_reg[2] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[5], weights_reg[2] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[6], weights_reg[2] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[7], weights_reg[2] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[8], weights_reg[2] );

			/**********************************************************************************************************
			  Second phase - multiply second row of weights  and second row of activations
			***********************************************************************************************************/

			 /* Load weights from SLM into registers - row1, output channels 0..7  */
				{
					 	__local uint *slm_ptrw1  = slm_ptr1 + 3*slm_read_pixel_offset;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1   			 += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1  			     += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
				}

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[9], weights_reg[0] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[10], weights_reg[0] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[11], weights_reg[0] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[12], weights_reg[0] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[13], weights_reg[0] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[14], weights_reg[0] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[15], weights_reg[0] );

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[10], weights_reg[1] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[11], weights_reg[1] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[12], weights_reg[1] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[13], weights_reg[1] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[14], weights_reg[1] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[15], weights_reg[1] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[16], weights_reg[1] );

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[11], weights_reg[2] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[12], weights_reg[2] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[13], weights_reg[2] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[14], weights_reg[2] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[15], weights_reg[2] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[16], weights_reg[2] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[17], weights_reg[2] );

				    /* Load weights from SLM into registers - row1, output channels 8..15  */
				{
					 	__local uint *slm_ptrw1 = slm_ptr1 + 3*slm_read_pixel_offset + 2*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1   			   += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1   			   += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
				}

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[9], weights_reg[0] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[10], weights_reg[0] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[11], weights_reg[0] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[12], weights_reg[0] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[13], weights_reg[0] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[14], weights_reg[0] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[15], weights_reg[0] );

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[10], weights_reg[1] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[11], weights_reg[1] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[12], weights_reg[1] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[13], weights_reg[1] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[14], weights_reg[1] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[15], weights_reg[1] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[16], weights_reg[1] );

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[11], weights_reg[2] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[12], weights_reg[2] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[13], weights_reg[2] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[14], weights_reg[2] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[15], weights_reg[2] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[16], weights_reg[2] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[17], weights_reg[2] );

				/* Load weights from SLM into registers - row1, output channels 16..23  */
				{
					 	__local uint *slm_ptrw1 = slm_ptr1 + 3*slm_read_pixel_offset + 4*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1   			   += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1   			   += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
				}

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[9], weights_reg[0] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[10], weights_reg[0] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[11], weights_reg[0] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[12], weights_reg[0] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[13], weights_reg[0] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[14], weights_reg[0] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[15], weights_reg[0] );

				/* load 1Hx9Wx4N inputs, Activation row2  */

				uint slm_row_offset_3	  = 2*(TILE_W + 2)*BATCH_PACK*8;

				__attribute__((opencl_unroll_hint(9)))
				for (int ic = 0; ic < 9; ic++)
				{
	                 /* Load activations from SLM into registers  */

					 uint slm_offset = slm_row_offset_3 + ic * BATCH_PACK * 8 ;

    				 act_reg [ ic ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ;
				}

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[10], weights_reg[1] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[11], weights_reg[1] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[12], weights_reg[1] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[13], weights_reg[1] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[14], weights_reg[1] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[15], weights_reg[1] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[16], weights_reg[1] );

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[11], weights_reg[2] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[12], weights_reg[2] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[13], weights_reg[2] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[14], weights_reg[2] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[15], weights_reg[2] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[16], weights_reg[2] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[17], weights_reg[2] );

				/* Load weights from SLM into registers - row1, output channels 24..31  */
				{
					 	__local uint *slm_ptrw1 = slm_ptr1 + 3*slm_read_pixel_offset + 6*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1   			   += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
						slm_ptrw1   			   += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw1 + 64 ) );
				}

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[9], weights_reg[0] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[10], weights_reg[0] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[11], weights_reg[0] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[12], weights_reg[0] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[13], weights_reg[0] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[14], weights_reg[0] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[15], weights_reg[0] );

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[10], weights_reg[1] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[11], weights_reg[1] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[12], weights_reg[1] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[13], weights_reg[1] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[14], weights_reg[1] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[15], weights_reg[1] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[16], weights_reg[1] );

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[11], weights_reg[2] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[12], weights_reg[2] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[13], weights_reg[2] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[14], weights_reg[2] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[15], weights_reg[2] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[16], weights_reg[2] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[17], weights_reg[2] );

			/**********************************************************************************************************
			  Third phase - multiply third row of weights  and third row of activations
			***********************************************************************************************************/

				 /* Load weights from SLM into registers - row2, output channels 0..7  */
				{
					 	__local uint *slm_ptrw2  = slm_ptr1 + 6*slm_read_pixel_offset;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2   			   += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2 			     += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
				}

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[0], weights_reg[0] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[1], weights_reg[0] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[2], weights_reg[0] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[3], weights_reg[0] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[4], weights_reg[0] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[5], weights_reg[0] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[6], weights_reg[0] );

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[1], weights_reg[1] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[2], weights_reg[1] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[3], weights_reg[1] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[4], weights_reg[1] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[5], weights_reg[1] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[6], weights_reg[1] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[7], weights_reg[1] );

				out_07[ 0 ] = _MMAD_4x8 ( out_07[ 0 ], act_reg[2], weights_reg[2] );
				out_07[ 1 ] = _MMAD_4x8 ( out_07[ 1 ], act_reg[3], weights_reg[2] );
				out_07[ 2 ] = _MMAD_4x8 ( out_07[ 2 ], act_reg[4], weights_reg[2] );
				out_07[ 3 ] = _MMAD_4x8 ( out_07[ 3 ], act_reg[5], weights_reg[2] );
				out_07[ 4 ] = _MMAD_4x8 ( out_07[ 4 ], act_reg[6], weights_reg[2] );
				out_07[ 5 ] = _MMAD_4x8 ( out_07[ 5 ], act_reg[7], weights_reg[2] );
				out_07[ 6 ] = _MMAD_4x8 ( out_07[ 6 ], act_reg[8], weights_reg[2] );

				     /* Load weights from SLM into registers - row2, output channels 8..15  */
				{
					 	__local uint *slm_ptrw2 = slm_ptr1 + 6*slm_read_pixel_offset + 2*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2   			   += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2   			   += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
				}

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[0], weights_reg[0] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[1], weights_reg[0] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[2], weights_reg[0] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[3], weights_reg[0] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[4], weights_reg[0] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[5], weights_reg[0] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[6], weights_reg[0] );

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[1], weights_reg[1] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[2], weights_reg[1] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[3], weights_reg[1] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[4], weights_reg[1] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[5], weights_reg[1] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[6], weights_reg[1] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[7], weights_reg[1] );

				out_815[ 0 ] = _MMAD_4x8 ( out_815[ 0 ], act_reg[2], weights_reg[2] );
				out_815[ 1 ] = _MMAD_4x8 ( out_815[ 1 ], act_reg[3], weights_reg[2] );
				out_815[ 2 ] = _MMAD_4x8 ( out_815[ 2 ], act_reg[4], weights_reg[2] );
				out_815[ 3 ] = _MMAD_4x8 ( out_815[ 3 ], act_reg[5], weights_reg[2] );
				out_815[ 4 ] = _MMAD_4x8 ( out_815[ 4 ], act_reg[6], weights_reg[2] );
				out_815[ 5 ] = _MMAD_4x8 ( out_815[ 5 ], act_reg[7], weights_reg[2] );
				out_815[ 6 ] = _MMAD_4x8 ( out_815[ 6 ], act_reg[8], weights_reg[2] );

				/* Load weights from SLM into registers - row2, output channels 16..23  */
				{
					 	__local uint *slm_ptrw2 = slm_ptr1 + 6*slm_read_pixel_offset + 4*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2   			   += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2   			   += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
				}

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[0], weights_reg[0] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[1], weights_reg[0] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[2], weights_reg[0] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[3], weights_reg[0] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[4], weights_reg[0] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[5], weights_reg[0] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[6], weights_reg[0] );

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[1], weights_reg[1] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[2], weights_reg[1] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[3], weights_reg[1] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[4], weights_reg[1] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[5], weights_reg[1] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[6], weights_reg[1] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[7], weights_reg[1] );

				out_1623[ 0 ] = _MMAD_4x8 ( out_1623[ 0 ], act_reg[2], weights_reg[2] );
				out_1623[ 1 ] = _MMAD_4x8 ( out_1623[ 1 ], act_reg[3], weights_reg[2] );
				out_1623[ 2 ] = _MMAD_4x8 ( out_1623[ 2 ], act_reg[4], weights_reg[2] );
				out_1623[ 3 ] = _MMAD_4x8 ( out_1623[ 3 ], act_reg[5], weights_reg[2] );
				out_1623[ 4 ] = _MMAD_4x8 ( out_1623[ 4 ], act_reg[6], weights_reg[2] );
				out_1623[ 5 ] = _MMAD_4x8 ( out_1623[ 5 ], act_reg[7], weights_reg[2] );
				out_1623[ 6 ] = _MMAD_4x8 ( out_1623[ 6 ], act_reg[8], weights_reg[2] );

				/* Load weights from SLM into registers - row3, output channels 24..31  */
				{
					 	__local uint *slm_ptrw2 = slm_ptr1 + 6*slm_read_pixel_offset + 6*8*8;

					    weights_reg[0].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[0].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2   			   += slm_read_pixel_offset;

						weights_reg[1].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[1].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
						slm_ptrw2   			   += slm_read_pixel_offset;

						weights_reg[2].s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 ) );
					    weights_reg[2].s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptrw2 + 64 ) );
				}

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[0], weights_reg[0] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[1], weights_reg[0] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[2], weights_reg[0] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[3], weights_reg[0] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[4], weights_reg[0] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[5], weights_reg[0] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[6], weights_reg[0] );

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[1], weights_reg[1] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[2], weights_reg[1] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[3], weights_reg[1] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[4], weights_reg[1] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[5], weights_reg[1] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[6], weights_reg[1] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[7], weights_reg[1] );

				out_2431[ 0 ] = _MMAD_4x8 ( out_2431[ 0 ], act_reg[2], weights_reg[2] );
				out_2431[ 1 ] = _MMAD_4x8 ( out_2431[ 1 ], act_reg[3], weights_reg[2] );
				out_2431[ 2 ] = _MMAD_4x8 ( out_2431[ 2 ], act_reg[4], weights_reg[2] );
				out_2431[ 3 ] = _MMAD_4x8 ( out_2431[ 3 ], act_reg[5], weights_reg[2] );
				out_2431[ 4 ] = _MMAD_4x8 ( out_2431[ 4 ], act_reg[6], weights_reg[2] );
				out_2431[ 5 ] = _MMAD_4x8 ( out_2431[ 5 ], act_reg[7], weights_reg[2] );
				out_2431[ 6 ] = _MMAD_4x8 ( out_2431[ 6 ], act_reg[8], weights_reg[2] );
		}

			// To make sure all threads in WG have finished compute before next depth tile of activation and weights are loaded into SLM
			barrier(CLK_LOCAL_MEM_FENCE);
	} //for kd

        /****************************************************************************************************************
		*******************************Output Write Stage****************************************************************
		****************************************************************************************************************/
			/*
		   Outputs will be passed through activation function and quantized to 8 bits before writing
		   Output will be in same format as input [K/32][N/4][P][Q][4N][32K] */

			/******************* Write output to SLM *************************************/

		/*  Quantize and pack 4x1 byte - from consectuive n-coordinates
			Each thread produces [1P][7Q][4N][32K]
         	Write uint32 from each lane to SLM , the entire thread will write 32-consecutive K-coorindates

			Assume one SLM row as 32 uints ( 32 channels , four batches for each channel - 4NK )
			In SLM 7x7x4x32 present first then the next 32 channels
		*/

		if( lid_z <= 6 )
        {
			/* feature maps are an array of slicePacks, each H,W position within the slice pack contains 32 8bit feature maps(channels) of 8 different batches */
			uint row_size_bytes        = (_OW + OWPAD) * PACK * BATCH_PACK;

			/* slice_pack is a pack of 32 feature map tiles that are [OH][OW][4][32] that are stored within the full [K/32][N/4][OH][OW][4][32] output */
			uint slice_pack_size_bytes = row_size_bytes * (_OH + OHPAD);

			/* Each output_depth WG writes 64 output channels */

		 	uint output_depth_index      =  output_depth*2 + threadid_mod_2;
			uint batch_index			 =  batch;

			/* Each WG produces entire 7x7 output, hence no group_y, group_z tiling */

            uint output_offset_x = groupy_tile * OUT_X_PITCH;
            uint output_offset_y = groupz_tile * OUT_Y_PITCH;
			uint slice_pack_addr_bytes  = output_depth_index * slice_pack_size_bytes * ( BATCH_SIZE / BATCH_PACK ) + batch_index * slice_pack_size_bytes + lid_z * row_size_bytes;
						
			__global uchar* output_write_ptr = (__global uchar *) &outputs [ slice_pack_addr_bytes + output_offset_x + output_offset_y ];

                const uint feature = output_depth_index * 32 + get_sub_group_local_id();

                const float4 quant_f = as_float4(intel_sub_group_block_read4((__global uint*) (quantizations + feature) ));
                const float4 bias_f = as_float4(intel_sub_group_block_read4((__global uint*) (biases + feature) ));
                const float4 calib_f = as_float4(intel_sub_group_block_read4((__global uint*) (calibrations + feature) ));

                __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
				for (int col = 0; col < OUT_BLOCK_WIDTH; col++)
                {

					int4 outvec0 = out_07[col];
					int4 outvec1 = out_815[col];
					int4 outvec2 = out_1623[col];
					int4 outvec3 = out_2431[col];

					/* Non-Linear Activation & Quantization code */

					uchar8 out_write_N2K4[2];

                    QUANTIZATION;

					intel_sub_group_block_write_uc8 (  output_write_ptr  , out_write_N2K4[0] );
					output_write_ptr += 64;
					intel_sub_group_block_write_uc8 (  output_write_ptr  , out_write_N2K4[1] );
					output_write_ptr += 64;

				} // out_block_width-for loop
		}//lid_z loop
} //end of kernel

#undef SCAL
#undef QUANTIZATION
