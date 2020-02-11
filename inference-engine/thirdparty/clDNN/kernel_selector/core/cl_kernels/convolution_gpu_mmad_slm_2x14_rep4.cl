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

#include "include/data_types.cl"
#include "include/mmad.cl"

#define SCALE 0.11f

#ifdef LIGHTWEIGHT_QUANTIZATION

#define QUANTIZATION \
    slm_write0.s0 = convert_uchar_sat((float)outvec.s0 * SCALE + bias_f);\
    slm_write0.s1 = convert_uchar_sat((float)outvec.s1 * SCALE + bias_f);\
    slm_write0.s2 = convert_uchar_sat((float)outvec.s2 * SCALE + bias_f);\
    slm_write0.s3 = convert_uchar_sat((float)outvec.s3 * SCALE + bias_f);

#elif NO_QUANTIZATION

#define QUANTIZATION(idx) \
    slm_write0.s0 = convert_uchar_sat(outvec.s0);\
    slm_write0.s1 = convert_uchar_sat(outvec.s1);\
    slm_write0.s2 = convert_uchar_sat(outvec.s2);\
    slm_write0.s3 = convert_uchar_sat(outvec.s3);

#else

#define QUANTIZATION \
    slm_write0.s0 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec.s0) * quant_f * I_QF + bias_f) * calib_f)), ACTIVATION_PARAMS));\
    slm_write0.s1 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec.s1) * quant_f * I_QF + bias_f) * calib_f)), ACTIVATION_PARAMS));\
    slm_write0.s2 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec.s2) * quant_f * I_QF + bias_f) * calib_f)), ACTIVATION_PARAMS));\
    slm_write0.s3 = as_uchar(ACTIVATION(convert_char(round(((float)(outvec.s3) * quant_f * I_QF + bias_f) * calib_f)), ACTIVATION_PARAMS));

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

	/* 32,1,4 WG , SIMD8 - 16 HW threads in a WG
	threads 0-3   (group1) : (lid_x:0-15,lid_y:0,lid_z:0)	
	threads 4-7   (group2) : (lid_x:0-15,lid_y:0,lid_z:1)
	threads 8-11  (group3) : (lid_x:0-15,lid_y:0,lid_z:2)
	threads 12-15  (group4) : (lid_x:0-15,lid_y:0,lid_z:3)
	
    Verify sub_group_layout through below printfs 
	
	if(group_z == 0 && group_y == 0 && fmg == 0 && get_sub_group_id() == 31) { 
			printf("\n sub_group_local_id: %d, lid_x: %d, lid_y: %d, lid_z: %d ", get_sub_group_local_id(), get_local_id(0) ,get_local_id(1),get_local_id(2));	
			printf("\n #WorkgroupsX: %d, #WorkgroupsY: %d, #WorkgroupsZ: %d",get_num_groups(0),get_num_groups(1),get_num_groups(2)); 	
	}
	
	If sub_group_layout is different then derive lid_x, lid_z
	
	lid_z: thread_id/4
	*/
	
	/* Thread, local IDs */
	ushort thread_id 		= get_sub_group_id();
	ushort threadid_group_4 = thread_id % 4;
	ushort threadid_mod_2   = thread_id%2;
	ushort threadid_mod_8   = thread_id % 8;

	ushort lid_x    = get_local_id(0);
	ushort lid_z    = get_local_id(2);

	uchar  lane_id  = get_sub_group_local_id();

	/* 32-bit signed accumulator for 4 mini-batches , for a thread OUT_BLOCK_WIDTH*HEIGHT*4 registers are used
	   Will be converted to 8-bits before final write														*/
	 
	int4 out[ OUT_BLOCK_HEIGHT * OUT_BLOCK_WIDTH ] = { 0 } ;
	
	/* Account for batching */

	ushort batch = ( fmg*LOCAL_SIZE_X ) /_OD;

	// Size calculated for int8 elements , One Batch processing is [H][W][4N][32C]
	uint input_size = (_IH + IHPAD) * (_IW + IWPAD) * BATCH_PACK ;  
	
	uint in_addr_offset = batch*input_size;
	
	/* Goto activation tile for work group, offset is w.r.t int8 array */
	
	uint groupy_tile = TILE_W*group_y;
	uint groupz_tile = TILE_H*group_z;
	
     in_addr_offset += (groupz_tile * K_STRIDE) * (_IW + IWPAD) * BATCH_PACK + (groupy_tile * K_STRIDE) * BATCH_PACK;
	 
	 	/* SLM space for Activation, Weights
	       ( 32,1,4 ) Workgroup - 4 tiles along Y direction and 32 different output channels
	        Activation - 10Wx16Wx4Nx32C Weights -9RSx32Kx32C	*/
	
	__local int8 act_slm      [  10*16*4 ];
	__local int8 weight_slm   [  9*32  ];
   
   /* 10Hx16Wx4Nx32C activation tile written into SLM.  Distribute among 16 threads in Workgroup
	   threads 0-1 write 16x4x32 of H=0, W=0...15 ( 8x4x32 per thread )
	   threads 2-3 write 16x4x32 of H=1, W=0...15 ( 8x4x32 per thread )
	   threads 4-5 write 16x4x32 of H=2, W=0...15 ( 8x4x32 per thread )
	   threads 6-7 write 16x4x32 of H=3, W=0...15 ( 8x4x32 per thread )
	   threads 8-9 write 16x4x32 of H=4, W=0...15 ( 8x4x32 per thread )
	   threads 10-11 write 16x4x32 of H=5, W=0...15 ( 8x4x32 per thread )
	   threads 12 write 16x4x32 of H=6, W=0...15 ( 16x4x32 per thread )
	   thread 13 writes 16x4x32 of H=7
	   thread 14 writes 16x4x32 of H=8
	   thread 15 writes 16x4x32 of H=9

	   Interleaved write to avoid SLM BC
	   
	   threads0,1 write 16x4x32 together
	   thread0 writes first 4x32 block, thread1 writes next 4x32 block etc.
   */

        
	/* Goto activation tile for thread in group */
	
	uint row_offset   =  thread_id / 2;
	
	if ( thread_id >= 12 ) {
		row_offset = 6 + thread_id - 12 - threadid_mod_2;
	}
	
	// In addr offset for the particular thread
	in_addr_offset    += row_offset * K_STRIDE * (_IW + IWPAD ) * BATCH_PACK ;

   /* Activation SLM indices */
    uint act_slm_write =  row_offset * ( TILE_W + 2) * BATCH_PACK;
	uint act_slm_read  =  OUT_BLOCK_HEIGHT * lid_z * ( TILE_W + 2) * BATCH_PACK ;

	/* Weights 
	   Weight Global Tensor Order: [K/8][C/32][R][S][8C][8K][4C]
	*/
	
	/* 9RSx32Kx32C Weight Block in SLM
	   thread0 handles ( reads from global ) w(0,0),w(0,1),w(0,2) of K=0 ( k=0..7)
	   thread1 handles w(0,0),w(0,1),w(0,2) of K=1 ( k=8..15)
	   thread2 handles w(1,0),w(1,1) of K=0 ( k=0..7)
	   thread3 handles w(1,0),w(1,1) of K=1 ( k=8..15)
	   thread4 handles w(1,2),w(2,0) of K=0 ( k=0..7)
	   thread5 handles w(1,2),w(2,0) of K=1 ( k=8..15)
	   thread6 handles w(2,1),w(2,2) of K=0 ( k=0..7)
	   thread7 handles w(2,1),w(2,2) of K=1 ( k=8..15)
	   
	   Similarly threads8-15 handles for K=2,3
	   
	   Weight Layout in SLM
	   
	   w(R=0,S=0,k=0..7,C=0..15),w(R=0,S=0,k=8..15,C=0..15)
	   w(R=0,S=0,k=0..7,C=16..31),w(R=0,S=0,k=8..15,C=16..31)
	   
	   Above interleaving present to avoid SLM Bank conflicts when fused threads read from SLM
	   Thread0 will read k=0..7, thread1 will read k=8..15
	   
	   First all output channels are present in SLM, then next weight pixel is present in SLM */
	  
	 #define NUM_FILTERS (K_HEIGHT * K_WIDTH)
	  
	 uint output_depth = fmg % ( _OD / LOCAL_SIZE_X ); 
	 	  
	 uint weight_size_CRS =  ( _ID / PACK ) * NUM_FILTERS * 8; //8 output channels packed inside
	 
	 // Global weight addr for workgroup
	 uint weight_global_addr_offset =  output_depth * 4 * weight_size_CRS ; //32 output channels per workgroup
	 
	 // Global weight address for thread 
	 uint weight_global_channel_offset = threadid_mod_2 * weight_size_CRS ;
	 
	uint slm_channel_offset = 0;
	 
    if ( thread_id >= 8 ) {
		weight_global_channel_offset +=  2*weight_size_CRS;
		slm_channel_offset = 1;	
    }
	 
	 uint weight_global_pixel_offset = 0;
	 uint slm_pixel_offset = 0;
	 
    if ( threadid_mod_8 >=2  )
    {
		weight_global_pixel_offset = 3*8 +  ( ( (threadid_mod_8/2) - 1 )*2*8 );
		slm_pixel_offset 		   = 3*LOCAL_SIZE_X + ( ( (threadid_mod_8/2) - 1 )*2*LOCAL_SIZE_X );
    }
	 
	 weight_global_addr_offset += weight_global_channel_offset + weight_global_pixel_offset;
	 
	 /* Weight slm write index */
	 
	 uint slm_write_weight = threadid_mod_2*4  + slm_pixel_offset + slm_channel_offset * 16;
	 
	 /* Weight slm read index */
	 
	 uint wt_slm_rd_offset = threadid_group_4*8;
 
    if ( threadid_mod_2 )
    {
		wt_slm_rd_offset = wt_slm_rd_offset - 8 + 4;
    }
	
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
					
			int4 act_col_0 =  as_int4( intel_sub_group_block_read4(activation_tile) );	
			int4 act_col_1 =  as_int4( intel_sub_group_block_read4(activation_tile + 8*8) );				
			int4 act_col_2 =  as_int4( intel_sub_group_block_read4(activation_tile + 2*8*8) );				
			int4 act_col_3 =  as_int4( intel_sub_group_block_read4(activation_tile + 3*8*8) );
			int4 act_col_4 =  as_int4( intel_sub_group_block_read4(activation_tile + 4*8*8) );			
			int4 act_col_5 =  as_int4( intel_sub_group_block_read4(activation_tile + 5*8*8) );								
			int4 act_col_6 =  as_int4( intel_sub_group_block_read4(activation_tile + 6*8*8) );				
			int4 act_col_7 =  as_int4( intel_sub_group_block_read4(activation_tile + 7*8*8) );				

			SLM_BLOCK_WRITE_4 ( act_slm_ptr , as_uint4 ( act_col_0 ) );				
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 8*8 ) , as_uint4 ( act_col_1 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 2*8*8 ) , as_uint4 ( act_col_2 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 3*8*8 ) , as_uint4 ( act_col_3 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 4*8*8 ) , as_uint4 ( act_col_4 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 5*8*8 ) , as_uint4 ( act_col_5 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 6*8*8 ) , as_uint4 ( act_col_6 ) );
			SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 7*8*8 ) , as_uint4 ( act_col_7 ) );

			if ( thread_id >=12 )
            {
				activation_tile = activation_tile + 1 * (_IW + IWPAD ) * BATCH_PACK * 8;
				act_slm_ptr 	+= 8*8*8;		
	
				int4 act_col_9 =  as_int4( intel_sub_group_block_read4(activation_tile) );				
				int4 act_col_10 =  as_int4( intel_sub_group_block_read4(activation_tile + 8*8) );				
				int4 act_col_11 =  as_int4( intel_sub_group_block_read4(activation_tile + 2*8*8) );				
				int4 act_col_12 =  as_int4( intel_sub_group_block_read4(activation_tile + 3*8*8) );
				int4 act_col_13 =  as_int4( intel_sub_group_block_read4(activation_tile + 4*8*8) );			
				int4 act_col_14 =  as_int4( intel_sub_group_block_read4(activation_tile + 5*8*8) );								
				int4 act_col_15 =  as_int4( intel_sub_group_block_read4(activation_tile + 6*8*8) );				
				int4 act_col_16 =  as_int4( intel_sub_group_block_read4(activation_tile + 7*8*8) );				
				
				SLM_BLOCK_WRITE_4 ( act_slm_ptr  , as_uint4 ( act_col_9 ) );				
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 8*8 )   , as_uint4 ( act_col_10 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 2*8*8 ) , as_uint4 ( act_col_11 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 3*8*8 ) , as_uint4 ( act_col_12 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 4*8*8 ) , as_uint4 ( act_col_13 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 5*8*8 ) , as_uint4 ( act_col_14 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 6*8*8 ) , as_uint4 ( act_col_15 ) );
				SLM_BLOCK_WRITE_4 ( ( act_slm_ptr + 7*8*8 ) , as_uint4 ( act_col_16 ) );
			}

		/* load weights from global to weight_slm */
		
			int weight_addr = kd * NUM_FILTERS * 8 + weight_global_addr_offset;

			__global uint *weight_tile   = (__global uint*)&weights    [ weight_addr ];
			__local  uint *wt_slm_ptr    = (__local uint *) &weight_slm [ slm_write_weight  ];
			
			int4 w0 = as_int4 ( intel_sub_group_block_read4( weight_tile ) );						
			int4 w1 = as_int4 ( intel_sub_group_block_read4( weight_tile + 4*8 ) );			
			int4 w2 = as_int4 ( intel_sub_group_block_read4( weight_tile + 8*8 ) );	
			int4 w3 = as_int4 ( intel_sub_group_block_read4( weight_tile + 12*8 ) );
			
			SLM_BLOCK_WRITE_4 ( wt_slm_ptr , as_uint4 ( w0 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 8*8 )   , as_uint4 ( w1 ) );		
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 32*8 ) , as_uint4 ( w2 ) );
			SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 32*8 + 8*8 ) , as_uint4 ( w3 ) );
		   
		   if( threadid_mod_8 < 2 )
           { 
				weight_tile += 16*8;
				wt_slm_ptr  += 2*32*8;
			
				int4 w4 = as_int4 ( intel_sub_group_block_read4( weight_tile ) );						
				int4 w5 = as_int4 ( intel_sub_group_block_read4( weight_tile + 4*8 ) );			
				
				SLM_BLOCK_WRITE_4 ( wt_slm_ptr , as_uint4 ( w4 ) );
				SLM_BLOCK_WRITE_4 ( ( wt_slm_ptr + 8*8 )   , as_uint4 ( w5 ) );		
			}
	}		

		// Synchronize SLM writes across workgroup
		 barrier(CLK_LOCAL_MEM_FENCE);		

			uint wt_slm_rd = wt_slm_rd_offset;
		
			__local uint *slm_ptr0     = (__local uint *) &act_slm[ act_slm_read ];
			__local uint *slm_ptr1     = (__local uint *) &weight_slm[ wt_slm_rd ];
			
			int8 weights_reg0, weights_reg1,weights_reg2;
			
			/**********************************************************************************************************
			  First phase - load first row of weights and for the first activation row - 1Hx8Wx4N inputs at a time
                          - Weights - 24 registers, Activations - 32 registers: Total 56 registers used	for input data			  
			***********************************************************************************************************/
			{ 
					int4 act_reg[ 8 ];
	
	                 /* Load weights from SLM into registers  */
				{
					    weights_reg0.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg0.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
						
						weights_reg1.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg1.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
						
						weights_reg2.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg2.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
				}
						
			/* load first 1Hx8Wx4N inputs - Activation Broadcast will occur since it is same for fused threads */
			
				__attribute__((opencl_unroll_hint(8)))
				for (int ic = 0; ic < 8; ic++)
				{
	                 /* Load activations from SLM into registers  */
					 
					 uint slm_offset = ic * BATCH_PACK * 8 ;
					 
    				 act_reg [ ic ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ; 
				}
			
			/* Convolve */ 
			
			   /* order the mmad instructions to minimize dependency on src0,dst - also try to maximise reuse of weights-reg*/

                out[ 0 ] = _MMAD_4x8 ( out[ 0 ], act_reg[0], weights_reg0 );
				out[ 1 ] = _MMAD_4x8 ( out[ 1 ], act_reg[1], weights_reg0 );
				out[ 2 ] = _MMAD_4x8 ( out[ 2 ], act_reg[2], weights_reg0 );
				out[ 3 ] = _MMAD_4x8 ( out[ 3 ], act_reg[3], weights_reg0 );
				out[ 4 ] = _MMAD_4x8 ( out[ 4 ], act_reg[4], weights_reg0 );
				out[ 5 ] = _MMAD_4x8 ( out[ 5 ], act_reg[5], weights_reg0 );
				out[ 6 ] = _MMAD_4x8 ( out[ 6 ], act_reg[6], weights_reg0 );
				out[ 7 ] = _MMAD_4x8 ( out[ 7 ], act_reg[7], weights_reg0 );

				out[ 0 ] = _MMAD_4x8 ( out[ 0 ], act_reg[1], weights_reg1 );
				out[ 1 ] = _MMAD_4x8 ( out[ 1 ], act_reg[2], weights_reg1 );
				out[ 2 ] = _MMAD_4x8 ( out[ 2 ], act_reg[3], weights_reg1 );
				out[ 3 ] = _MMAD_4x8 ( out[ 3 ], act_reg[4], weights_reg1 );
				out[ 4 ] = _MMAD_4x8 ( out[ 4 ], act_reg[5], weights_reg1 );
				out[ 5 ] = _MMAD_4x8 ( out[ 5 ], act_reg[6], weights_reg1 );
				out[ 6 ] = _MMAD_4x8 ( out[ 6 ], act_reg[7], weights_reg1 );
				
				out[ 0 ] = _MMAD_4x8 ( out[ 0 ], act_reg[2], weights_reg2 );
				out[ 1 ] = _MMAD_4x8 ( out[ 1 ], act_reg[3], weights_reg2 );
				out[ 2 ] = _MMAD_4x8 ( out[ 2 ], act_reg[4], weights_reg2 );
				out[ 3 ] = _MMAD_4x8 ( out[ 3 ], act_reg[5], weights_reg2 );
				out[ 4 ] = _MMAD_4x8 ( out[ 4 ], act_reg[6], weights_reg2 );
				out[ 5 ] = _MMAD_4x8 ( out[ 5 ], act_reg[7], weights_reg2 );
			   
				/* load next 1Hx8Wx4N inputs */
		
				__attribute__((opencl_unroll_hint(8)))
				for (int ic = 8; ic < 16; ic++)
				{
					 uint slm_offset = ic * BATCH_PACK * 8;
					 
					 act_reg [ ic - 8 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset) ) ; 
				}
				
				/* Convolve */				
				
				out[ 6 ] = _MMAD_4x8 ( out[ 6 ], act_reg[0], weights_reg2 );
				out[ 7 ] = _MMAD_4x8 ( out[ 7 ], act_reg[1], weights_reg2 );
				out[ 8 ] = _MMAD_4x8 ( out[ 8 ], act_reg[2], weights_reg2 );
				out[ 9 ] = _MMAD_4x8 ( out[ 9 ], act_reg[3], weights_reg2 );
				out[ 10 ] = _MMAD_4x8 ( out[ 10 ], act_reg[4], weights_reg2 );
				out[ 11 ] = _MMAD_4x8 ( out[ 11 ], act_reg[5], weights_reg2 );
				out[ 12 ] = _MMAD_4x8 ( out[ 12 ], act_reg[6], weights_reg2 );
				out[ 13 ] = _MMAD_4x8 ( out[ 13 ], act_reg[7], weights_reg2 );
				
				out[ 7 ]  =  _MMAD_4x8 ( out[ 7 ], act_reg[0], weights_reg1 );
				out[ 8 ]  =  _MMAD_4x8 ( out[ 8 ], act_reg[1], weights_reg1 );
				out[ 9 ]  = _MMAD_4x8 (  out[ 9 ],  act_reg[2], weights_reg1 );
				out[ 10 ] = _MMAD_4x8 ( out[ 10 ], act_reg[3], weights_reg1 );
				out[ 11 ] = _MMAD_4x8 ( out[ 11 ], act_reg[4], weights_reg1 );
				out[ 12 ] = _MMAD_4x8 ( out[ 12 ], act_reg[5], weights_reg1 );
				out[ 13 ] = _MMAD_4x8 ( out[ 13 ], act_reg[6], weights_reg1 );
				
				out[ 8 ] =  _MMAD_4x8 ( out[ 8 ],  act_reg[0], weights_reg0 );
				out[ 9 ] = _MMAD_4x8 ( out [ 9 ],   act_reg[1], weights_reg0 );
				out[ 10 ] = _MMAD_4x8 ( out[ 10 ], act_reg[2], weights_reg0 );
				out[ 11 ] = _MMAD_4x8 ( out[ 11 ], act_reg[3], weights_reg0 );
				out[ 12 ] = _MMAD_4x8 ( out[ 12 ], act_reg[4], weights_reg0 );
				out[ 13 ] = _MMAD_4x8 ( out[ 13 ], act_reg[5], weights_reg0 );
			}	
			
			/* Second , Third phase */
		{
				int8 weights_reg3, weights_reg4,weights_reg5;
				int4 act_reg_2[ 6 ];

				/*****************************************************************************************************************************************
				 Second phase - load second row of weights, now both rows are in registers, for the second activation row - 1Hx6Wx4N inputs at a time
                              - Weights - 48 registers, Activations - 24 registers: Total 72 registers used	for input data			 
				******************************************************************************************************************************************/
				
				 /* Load weights of row = 1 from SLM into registers  */
				 {
				 
						weights_reg3.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg3.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
						
						weights_reg4.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg4.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
						
						weights_reg5.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg5.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
				}
				
				/* load input row =1,col=0:1  1Hx2Wx8N  */
					 
				uint slm_row_offset_2 	  = 1*(TILE_W + 2)*BATCH_PACK*8;	 
				
				act_reg_2 [ 0 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_2) ) ; 
				act_reg_2 [ 1 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_2 + BATCH_PACK*8) ) ; 
				
				out[ 14 ] = _MMAD_4x8 ( out[ 14 ], act_reg_2[0] , weights_reg0 );
				out[ 0 ]  = _MMAD_4x8 ( out[ 0 ],  act_reg_2[0]  , weights_reg3 );
				out[ 1 ]  = _MMAD_4x8 ( out[ 1 ],  act_reg_2[1]  , weights_reg3 );
				out[ 15 ] = _MMAD_4x8 ( out[ 15 ], act_reg_2[1] , weights_reg0 );
				
				out[ 14 ] = _MMAD_4x8 ( out[ 14 ], act_reg_2[1], weights_reg1 );
				out[ 0 ]  = _MMAD_4x8 ( out[ 0 ],  act_reg_2[1], weights_reg4 );			
				
				/* load input row =1,col=2:7,8:13,1Hx6Wx4N  */
				
				uint col = 2;
				
				__attribute__((opencl_unroll_hint(2)))
				do {
				
				uint slm_offset 	  = 1*(TILE_W + 2)*BATCH_PACK*8 + col*BATCH_PACK*8;	 
	
				act_reg_2 [ 0 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ; 
				act_reg_2 [ 1 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset +   BATCH_PACK*8)) ; 
				act_reg_2 [ 2 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 2*BATCH_PACK*8)) ; 
				act_reg_2 [ 3 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 3*BATCH_PACK*8) ) ; 
   				act_reg_2 [ 4 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 4*BATCH_PACK*8) ) ; 
   				act_reg_2 [ 5 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 5*BATCH_PACK*8) ) ; 
   
   				uint first_row_offset   = col - 2;
				uint second_row_offset  = 14 + col - 2;

   				out [ first_row_offset ]      =  _MMAD_4x8 ( out[ first_row_offset ] ,    act_reg_2[0] , weights_reg5 );
				out [ first_row_offset + 1 ]  =  _MMAD_4x8 ( out[ first_row_offset + 1] , act_reg_2[0],  weights_reg4 );
				out [ first_row_offset + 2 ]  =  _MMAD_4x8 ( out[ first_row_offset + 2] , act_reg_2[0],  weights_reg3 );
				out [ first_row_offset + 3 ]  =  _MMAD_4x8 ( out[ first_row_offset + 3 ], act_reg_2[1], weights_reg3 );
				
				out [ second_row_offset ]      =  _MMAD_4x8 ( out[ second_row_offset ] , act_reg_2[0] , weights_reg2 );			
				out [ second_row_offset + 1 ]  =  _MMAD_4x8 ( out[ second_row_offset + 1] , act_reg_2[0],  weights_reg1 );
				out [ second_row_offset + 2 ]  =  _MMAD_4x8 ( out[ second_row_offset + 2] , act_reg_2[0],  weights_reg0 );
				out [ second_row_offset + 3 ]  =  _MMAD_4x8 ( out[ second_row_offset + 3], act_reg_2[1], weights_reg0 );
				
				out [ first_row_offset + 1 ]   = _MMAD_4x8 (  out[ first_row_offset + 1 ], act_reg_2[1], weights_reg5 );
				out [ first_row_offset + 2 ]   = _MMAD_4x8 (  out[ first_row_offset + 2 ], act_reg_2[1], weights_reg4 );
				out [ first_row_offset + 3 ]   = _MMAD_4x8 ( out[ first_row_offset + 3 ],  act_reg_2[2], weights_reg4 );
				out [ first_row_offset + 4 ]   = _MMAD_4x8 ( out[ first_row_offset + 4 ],  act_reg_2[2], weights_reg3 );
				
				out [ second_row_offset + 1 ]  = _MMAD_4x8 (  out[ second_row_offset + 1 ], act_reg_2[1], weights_reg2 );
				out [ second_row_offset + 2 ]  = _MMAD_4x8 (  out[ second_row_offset + 2 ], act_reg_2[1], weights_reg1 );
				out [ second_row_offset + 3 ]   = _MMAD_4x8 ( out[ second_row_offset + 3 ], act_reg_2[2], weights_reg1 );
				out [ second_row_offset + 4 ]   = _MMAD_4x8 ( out[ second_row_offset + 4 ], act_reg_2[2], weights_reg0 );

				out [ first_row_offset + 2 ]   = _MMAD_4x8 ( out[ first_row_offset + 2], act_reg_2[2], weights_reg5 );				
				out [ first_row_offset + 3 ]   = _MMAD_4x8 ( out[ first_row_offset + 3], act_reg_2[3], weights_reg5 );
				out [ first_row_offset + 4 ]   = _MMAD_4x8 ( out[ first_row_offset + 4], act_reg_2[3], weights_reg4 );
				out [ first_row_offset + 5 ]   = _MMAD_4x8 ( out[ first_row_offset + 5], act_reg_2[3], weights_reg3 );	
				
				out [ second_row_offset + 2 ]   = _MMAD_4x8 ( out[ second_row_offset + 2], act_reg_2[2], weights_reg2 );				
				out [ second_row_offset + 3 ]   = _MMAD_4x8 ( out[ second_row_offset + 3], act_reg_2[3], weights_reg2 );
				out [ second_row_offset + 4 ]   = _MMAD_4x8 ( out[ second_row_offset + 4], act_reg_2[3], weights_reg1 );
				out [ second_row_offset + 5 ]   = _MMAD_4x8 ( out[ second_row_offset + 5], act_reg_2[3], weights_reg0 );	

				out [ first_row_offset + 6 ]   = _MMAD_4x8 ( out[ first_row_offset + 6], act_reg_2[4], weights_reg3 );
				out [ first_row_offset + 7 ]   = _MMAD_4x8 ( out[ first_row_offset + 7], act_reg_2[5], weights_reg3 );
				out [ first_row_offset + 5 ]   = _MMAD_4x8 ( out[ first_row_offset + 5], act_reg_2[4], weights_reg4 );
				out [ first_row_offset + 6 ]   = _MMAD_4x8 ( out[ first_row_offset + 6], act_reg_2[5], weights_reg4 );
				out [ first_row_offset + 4 ]   = _MMAD_4x8 ( out[ first_row_offset + 4], act_reg_2[4], weights_reg5 );
				out [ first_row_offset + 5 ]   = _MMAD_4x8 ( out[ first_row_offset + 5], act_reg_2[5], weights_reg5 );
				
				out [ second_row_offset + 6 ]   = _MMAD_4x8 ( out[ second_row_offset + 6], act_reg_2[4], weights_reg0 );
				out [ second_row_offset + 7 ]   = _MMAD_4x8 ( out[ second_row_offset + 7], act_reg_2[5], weights_reg0 );
				out [ second_row_offset + 5 ]   = _MMAD_4x8 ( out[ second_row_offset + 5], act_reg_2[4], weights_reg1 );
				out [ second_row_offset + 6 ]   = _MMAD_4x8 ( out[ second_row_offset + 6], act_reg_2[5], weights_reg1 );
				out [ second_row_offset + 4 ]   = _MMAD_4x8 ( out[ second_row_offset + 4], act_reg_2[4], weights_reg2 );				
				out [ second_row_offset + 5 ]   = _MMAD_4x8 ( out[ second_row_offset + 5], act_reg_2[5], weights_reg2 );

				col +=6;
				
				} while ( col < 14 );
				
				/* load input row =1,col=14:15  1Hx2Wx4N  */

				uint slm_row_offset_3 	  = 1 * (TILE_W + 2) * BATCH_PACK * 8 + 14 * BATCH_PACK * 8;	

				act_reg_2 [ 0 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_3)) ; 
				act_reg_2 [ 1 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_3 +   BATCH_PACK*8)) ; 
				
				out[ 13 ]  = _MMAD_4x8 ( out[ 13 ],   act_reg_2[0],  weights_reg4 );		
				out[ 27 ]  = _MMAD_4x8 ( out[ 27 ],   act_reg_2[0],  weights_reg1 );		
				out[ 26 ]  = _MMAD_4x8 ( out[ 26 ],   act_reg_2[0],  weights_reg2 );	
				
				out[ 12 ]  = _MMAD_4x8 ( out[ 12 ],  act_reg_2[0],  weights_reg5 );					
				out[ 13 ]  = _MMAD_4x8 ( out[ 13 ],  act_reg_2[1],  weights_reg5 );	
				
				out[ 27 ]  = _MMAD_4x8 ( out[ 27 ],  act_reg_2[1],  weights_reg2 );
				
                /****************************************************************************************************************************************
				   Third phase - load third row of weights, this replaces first weight row, for the third activation row read 1Hx6Wx4N inputs at a time 
				               - Weights - 48 registers, Activations - 24 registers: Total 72 registers used for input data			  
				*****************************************************************************************************************************************/
				
				 /* Load weights of row = 2 from SLM into registers - replaces row = 0 weights  */
				 {
					    weights_reg0.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg0.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
						
						weights_reg1.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg1.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
						
						weights_reg2.s0123     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 ) );
					    weights_reg2.s4567     = as_int4 ( SLM_BLOCK_READ_4 ( slm_ptr1 + 64 ) );
						slm_ptr1   			   += LOCAL_SIZE_X*8;	
				}
				
				uint slm_row_offset_4 	  = 2*(TILE_W + 2)*BATCH_PACK*8;	 
				
				act_reg_2 [ 0 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_4)) ; 
				act_reg_2 [ 1 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_4 + BATCH_PACK*8)) ; 
		
				out[ 14 ] = _MMAD_4x8 ( out[ 14 ], act_reg_2[0] , weights_reg3 );
				out[ 0 ]  = _MMAD_4x8 ( out[ 0 ],  act_reg_2[0]  , weights_reg0 );
				out[ 1 ]  = _MMAD_4x8 ( out[ 1 ],  act_reg_2[1]  , weights_reg0 );
				out[ 15 ] = _MMAD_4x8 ( out[ 15 ], act_reg_2[1] , weights_reg3 );
				
				out[ 14 ] = _MMAD_4x8 ( out[ 14 ], act_reg_2[1], weights_reg4 );
				out[ 0 ]  = _MMAD_4x8 ( out[ 0 ],  act_reg_2[1], weights_reg1 );	
				
				/* load input row =2,col=2:7,8:13,1Hx6Wx4N  */
				
				uint col_2 = 2;
				
				__attribute__((opencl_unroll_hint(2)))
				do {
				
				uint slm_offset 	  = 2*(TILE_W + 2)*BATCH_PACK*8 + col_2*BATCH_PACK*8;	 
	
				act_reg_2 [ 0 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ; 
				act_reg_2 [ 1 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset +   BATCH_PACK*8)) ; 
				act_reg_2 [ 2 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 2*BATCH_PACK*8)) ; 
				act_reg_2 [ 3 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 3*BATCH_PACK*8) ) ; 
   				act_reg_2 [ 4 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 4*BATCH_PACK*8) ) ; 
   				act_reg_2 [ 5 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset + 5*BATCH_PACK*8) ) ; 
   
   				uint first_row_offset   = col_2 - 2;
				uint second_row_offset  = 14 + col_2 - 2;
   			
				out [ first_row_offset + 1 ]  =  _MMAD_4x8 ( out[ first_row_offset + 1] , act_reg_2[0],  weights_reg1 );
				out [ first_row_offset + 2 ]  =  _MMAD_4x8 ( out[ first_row_offset + 2] , act_reg_2[0],  weights_reg0 );
				out [ first_row_offset + 3 ]  =  _MMAD_4x8 ( out[ first_row_offset + 3 ], act_reg_2[1], weights_reg0 );
				out [ first_row_offset ]      =  _MMAD_4x8 ( out[ first_row_offset ] ,    act_reg_2[0] , weights_reg2 );
				
				out [ second_row_offset + 1 ]  =  _MMAD_4x8 ( out[ second_row_offset + 1] , act_reg_2[0],  weights_reg4 );
				out [ second_row_offset + 2 ]  =  _MMAD_4x8 ( out[ second_row_offset + 2] , act_reg_2[0],  weights_reg3 );
				out [ second_row_offset + 3 ]  =  _MMAD_4x8 ( out[ second_row_offset + 3], act_reg_2[1], weights_reg3 );
				out [ second_row_offset ]      =  _MMAD_4x8 ( out[ second_row_offset ] , act_reg_2[0] , weights_reg5 );
				
				out [ first_row_offset + 1 ]   = _MMAD_4x8 (  out[ first_row_offset + 1 ], act_reg_2[1], weights_reg2 );
				out [ first_row_offset + 2 ]   = _MMAD_4x8 (  out[ first_row_offset + 2 ], act_reg_2[1], weights_reg1 );
				out [ first_row_offset + 3 ]   = _MMAD_4x8 ( out[ first_row_offset + 3 ],  act_reg_2[2], weights_reg1 );
				out [ first_row_offset + 4 ]   = _MMAD_4x8 ( out[ first_row_offset + 4 ],  act_reg_2[2], weights_reg0 );
				
				out [ second_row_offset + 1 ]  = _MMAD_4x8 (  out[ second_row_offset + 1 ], act_reg_2[1], weights_reg5 );
				out [ second_row_offset + 2 ]  = _MMAD_4x8 (  out[ second_row_offset + 2 ], act_reg_2[1], weights_reg4 );
				out [ second_row_offset + 3 ]   = _MMAD_4x8 ( out[ second_row_offset + 3 ], act_reg_2[2], weights_reg4 );
				out [ second_row_offset + 4 ]   = _MMAD_4x8 ( out[ second_row_offset + 4 ], act_reg_2[2], weights_reg3 );

				out [ first_row_offset + 5 ]   = _MMAD_4x8 ( out[ first_row_offset + 5], act_reg_2[3], weights_reg0 );	
				out [ first_row_offset + 2 ]   = _MMAD_4x8 ( out[ first_row_offset + 2], act_reg_2[2], weights_reg2 );				
				out [ first_row_offset + 3 ]   = _MMAD_4x8 ( out[ first_row_offset + 3], act_reg_2[3], weights_reg2 );
				out [ first_row_offset + 4 ]   = _MMAD_4x8 ( out[ first_row_offset + 4], act_reg_2[3], weights_reg1 );
				
				out [ second_row_offset + 5 ]   = _MMAD_4x8 ( out[ second_row_offset + 5], act_reg_2[3], weights_reg3 );	
				out [ second_row_offset + 2 ]   = _MMAD_4x8 ( out[ second_row_offset + 2], act_reg_2[2], weights_reg5 );				
				out [ second_row_offset + 3 ]   = _MMAD_4x8 ( out[ second_row_offset + 3], act_reg_2[3], weights_reg5 );
				out [ second_row_offset + 4 ]   = _MMAD_4x8 ( out[ second_row_offset + 4], act_reg_2[3], weights_reg4 );

				out [ first_row_offset + 6 ]   = _MMAD_4x8 ( out[ first_row_offset + 6], act_reg_2[4], weights_reg0 );
				out [ first_row_offset + 7 ]   = _MMAD_4x8 ( out[ first_row_offset + 7], act_reg_2[5], weights_reg0 );
				out [ first_row_offset + 5 ]   = _MMAD_4x8 ( out[ first_row_offset + 5], act_reg_2[4], weights_reg1 );
				out [ first_row_offset + 6 ]   = _MMAD_4x8 ( out[ first_row_offset + 6], act_reg_2[5], weights_reg1 );				
				out [ first_row_offset + 4 ]   = _MMAD_4x8 ( out[ first_row_offset + 4], act_reg_2[4], weights_reg2 );
				out [ first_row_offset + 5 ]   = _MMAD_4x8 ( out[ first_row_offset + 5], act_reg_2[5], weights_reg2 );
				
				out [ second_row_offset + 6 ]   = _MMAD_4x8 ( out[ second_row_offset + 6], act_reg_2[4], weights_reg3 );
				out [ second_row_offset + 7 ]   = _MMAD_4x8 ( out[ second_row_offset + 7], act_reg_2[5], weights_reg3 );
				out [ second_row_offset + 5 ]   = _MMAD_4x8 ( out[ second_row_offset + 5], act_reg_2[4], weights_reg4 );
				out [ second_row_offset + 6 ]   = _MMAD_4x8 ( out[ second_row_offset + 6], act_reg_2[5], weights_reg4 );
				out [ second_row_offset + 4 ]   = _MMAD_4x8 ( out[ second_row_offset + 4], act_reg_2[4], weights_reg5 );				
				out [ second_row_offset + 5 ]   = _MMAD_4x8 ( out[ second_row_offset + 5], act_reg_2[5], weights_reg5 );
				
				col_2 +=6;
				
				} while ( col_2 < 14 );
				
				/* load input row =2,col=14:15  1Hx2Wx4N  */

				uint slm_row_offset_5 	  = 2 * (TILE_W + 2) * BATCH_PACK * 8 + 14 * BATCH_PACK * 8;	

				act_reg_2 [ 0 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_5)) ; 
				act_reg_2 [ 1 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_row_offset_5 +   BATCH_PACK*8)) ; 
				
				out[ 13 ]  = _MMAD_4x8 ( out[ 13 ],   act_reg_2[0],  weights_reg1 );		
				out[ 27 ]  = _MMAD_4x8 ( out[ 27 ],   act_reg_2[0],  weights_reg4 );		
				out[ 26 ]  = _MMAD_4x8 ( out[ 26 ],   act_reg_2[0],  weights_reg5 );	
				
				out[ 12 ]  = _MMAD_4x8 ( out[ 12 ],  act_reg_2[0],  weights_reg2 );					
				out[ 13 ]  = _MMAD_4x8 ( out[ 13 ],  act_reg_2[1],  weights_reg2 );	
				
				out[ 27 ]  = _MMAD_4x8 ( out[ 27 ],  act_reg_2[1],  weights_reg5 );
	}
				
				/*************************************************************************************************
				   Fourth phase - discard middle weight row, for fourth activation row load 1Hx8Wx4N at a time 
				                - Weights - 24 registers, Activations - 32 registers: Total 56 registers used for input data			  
				**************************************************************************************************/
		{ 
					int4 act_reg[ 8 ];
				
			/* load first 1Hx8Wx4N inputs */
			
				uint slm_row_offset_6 =  3 * (TILE_W + 2) * BATCH_PACK * 8 ;

				__attribute__((opencl_unroll_hint(8)))
				for (int ic = 0; ic < 8; ic++)
				{
	                 /* Load activations from SLM into registers  */
					 uint slm_offset = ic * BATCH_PACK * 8  + slm_row_offset_6;
    				 act_reg [ ic ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ; 
				}
			
			/* Convolve */ 
			
				uint phase_offset = 14;
				
				out[ phase_offset + 0 ] = _MMAD_4x8 ( out[ phase_offset +0 ], act_reg[0], weights_reg0 );
				out[ phase_offset + 1 ] = _MMAD_4x8 ( out[ phase_offset +1 ], act_reg[1], weights_reg0 );
				out[ phase_offset +2 ] = _MMAD_4x8 ( out[ phase_offset +2 ], act_reg[2], weights_reg0 );
				out[ phase_offset +3 ] = _MMAD_4x8 ( out[ phase_offset +3 ], act_reg[3], weights_reg0 );
				out[ phase_offset +4 ] = _MMAD_4x8 ( out[ phase_offset +4 ], act_reg[4], weights_reg0 );
				out[ phase_offset +5 ] = _MMAD_4x8 ( out[ phase_offset +5 ], act_reg[5], weights_reg0 );
				out[ phase_offset +6 ] = _MMAD_4x8 ( out[ phase_offset +6 ], act_reg[6], weights_reg0 );
				out[ phase_offset +7 ] = _MMAD_4x8 ( out[ phase_offset +7 ], act_reg[7], weights_reg0 );

				out[ phase_offset +0 ] = _MMAD_4x8 ( out[ phase_offset +0 ], act_reg[1], weights_reg1 );
				out[ phase_offset +1 ] = _MMAD_4x8 ( out[ phase_offset +1 ], act_reg[2], weights_reg1 );
				out[ phase_offset +2 ] = _MMAD_4x8 ( out[ phase_offset +2 ], act_reg[3], weights_reg1 );
				out[ phase_offset +3 ] = _MMAD_4x8 ( out[ phase_offset +3 ], act_reg[4], weights_reg1 );
				out[ phase_offset +4 ] = _MMAD_4x8 ( out[ phase_offset +4 ], act_reg[5], weights_reg1 );
				out[ phase_offset +5 ] = _MMAD_4x8 ( out[ phase_offset +5 ], act_reg[6], weights_reg1 );
				out[ phase_offset +6 ] = _MMAD_4x8 ( out[ phase_offset +6 ], act_reg[7], weights_reg1 );
				
				out[ phase_offset +0 ] = _MMAD_4x8 ( out[ phase_offset +0 ], act_reg[2], weights_reg2 );
				out[ phase_offset +1 ] = _MMAD_4x8 ( out[ phase_offset +1 ], act_reg[3], weights_reg2 );
				out[ phase_offset +2 ] = _MMAD_4x8 ( out[ phase_offset +2 ], act_reg[4], weights_reg2 );
				out[ phase_offset +3 ] = _MMAD_4x8 ( out[ phase_offset +3 ], act_reg[5], weights_reg2 );
				out[ phase_offset +4 ] = _MMAD_4x8 ( out[ phase_offset +4 ], act_reg[6], weights_reg2 );
				out[ phase_offset +5 ] = _MMAD_4x8 ( out[ phase_offset +5 ], act_reg[7], weights_reg2 );
			   
				/* load next 1Hx8Wx4N inputs */
		
				__attribute__((opencl_unroll_hint(8)))
				for (int ic = 8; ic < 16; ic++)
				{
					 uint slm_offset = ic * BATCH_PACK * 8 + slm_row_offset_6;
					 act_reg [ ic - 8 ] = as_int4 (SLM_BLOCK_READ_4 (slm_ptr0 + slm_offset)) ; 
				}
				
				/* Convolve */				
				
				out[ phase_offset +6 ] = _MMAD_4x8 ( out[ phase_offset +6 ], act_reg[0], weights_reg2 );
				out[ phase_offset +7 ] = _MMAD_4x8 ( out[ phase_offset +7 ], act_reg[1], weights_reg2 );
				out[ phase_offset + 8 ] = _MMAD_4x8 ( out[ phase_offset +8 ], act_reg[2], weights_reg2 );
				out[ phase_offset +9 ] = _MMAD_4x8 ( out[phase_offset + 9 ], act_reg[3], weights_reg2 );
				out[ phase_offset +10 ] = _MMAD_4x8 ( out[ phase_offset +10 ], act_reg[4], weights_reg2 );
				out[ phase_offset +11 ] = _MMAD_4x8 ( out[phase_offset + 11 ], act_reg[5], weights_reg2 );
				out[ phase_offset +12 ] = _MMAD_4x8 ( out[ phase_offset +12 ], act_reg[6], weights_reg2 );
				out[ phase_offset +13 ] = _MMAD_4x8 ( out[ phase_offset +13 ], act_reg[7], weights_reg2 );
				
				out[ phase_offset +7 ] =  _MMAD_4x8 ( out[ phase_offset +7 ], act_reg[0], weights_reg1 );
				out[ phase_offset +8 ] =  _MMAD_4x8 ( out[phase_offset + 8 ], act_reg[1], weights_reg1 );
				out[ phase_offset +9 ] = _MMAD_4x8 ( out[ phase_offset +9 ], act_reg[2], weights_reg1 );
				out[ phase_offset +10 ] = _MMAD_4x8 ( out[ phase_offset +10 ], act_reg[3], weights_reg1 );
				out[ phase_offset +11 ] = _MMAD_4x8 ( out[ phase_offset +11 ], act_reg[4], weights_reg1 );
				out[ phase_offset +12 ] = _MMAD_4x8 ( out[ phase_offset +12 ], act_reg[5], weights_reg1 );
				out[ phase_offset +13 ] = _MMAD_4x8 ( out[phase_offset + 13 ], act_reg[6], weights_reg1 );
				
				out[ phase_offset +8 ] =  _MMAD_4x8 ( out[phase_offset + 8 ],  act_reg[0], weights_reg0 );
				out[ phase_offset +9 ] = _MMAD_4x8 ( out[ phase_offset +9 ], act_reg[1], weights_reg0 );
				out[ phase_offset +10 ] = _MMAD_4x8 ( out[ phase_offset +10 ], act_reg[2], weights_reg0 );
				out[ phase_offset +11 ] = _MMAD_4x8 ( out[phase_offset + 11 ], act_reg[3], weights_reg0 );
				out[ phase_offset +12 ] = _MMAD_4x8 ( out[ phase_offset +12 ], act_reg[4], weights_reg0 );
				out[ phase_offset +13 ] = _MMAD_4x8 ( out[phase_offset + 13 ], act_reg[5], weights_reg0 );
			}	
			
			// To make sure all threads in WG have finished compute before next depth tile of activation and weights are loaded into SLM
			barrier(CLK_LOCAL_MEM_FENCE);	
			
	} //for kd

        /****************************************************************************************************************
		*******************************Output Write Stage****************************************************************
		****************************************************************************************************************/
		
		/* 
		   Outputs will be passed through activation function and quantized to 8 bits before writing 
		   Output will be in same format as input [K/32][N/4][P][Q][4N][32K]
		   Writes are staged in SLM so that 32-bit writes can be done to Global memory 
		*/	
			
		/******************* Write output to SLM *************************************/	
			
		/*  Quantize and pack 4x1 byte - from consectuive n-coordinates
         	Write uint32 from each lane to SLM , the entire thread will write 8-consecutive K-coorindates	
			Four threads will write 4x8xuint32 for 32 output channels and 4 batches
			This will be repeated for entire WG-tile
			
			Assume one SLM row as 32 uints ( 32 channels , four batches for each channel - 4NK )
		*/

			 uint out_slm_write        =  lid_z * TILE_W * OUT_BLOCK_HEIGHT * 32 + threadid_group_4 * 8 + lane_id;

			__local uchar4*  out_slm   = (__local uchar4*)  &act_slm;
			__local uchar4* out_slm_2  = (__local uchar4*)  &out_slm[ out_slm_write ];
		
			/* Scale the accumulator down and do the ReLU before converting to 8 bits */

			/*  Real code might do this, but need to get scale right or the convert to uchar saturates and then doesn''t match CPU 
			float scale = (float)SCALE_FACTOR;

			uchar outchar = (uchar)max(((float)outint) * scale, 0.0f); */

            const uint _feature = ((fmg * 32) % _OD) + (uint)get_local_id(0);
            float quant_f = as_float(intel_sub_group_block_read((__global uint*) (quantizations + _feature) ));
            float bias_f = as_float(intel_sub_group_block_read((__global uint*) (biases + _feature) ));
            float calib_f = as_float(intel_sub_group_block_read((__global uint*) (calibrations + _feature) ));

			__attribute__((opencl_unroll_hint(OUT_BLOCK_HEIGHT)))
			for (int r = 0; r < OUT_BLOCK_HEIGHT; r++)
            {
			    __attribute__((opencl_unroll_hint(OUT_BLOCK_WIDTH)))
				for (int c = 0; c < OUT_BLOCK_WIDTH; c++)
                {
					int4 outvec = out[ r * OUT_BLOCK_WIDTH + c];
			
					uchar4 slm_write0;
					
					int slm_addr = c * 32 + r * TILE_W * 32;
					
					/*TODO - Activation & Quantization  code goes here -  presently applying ReLU and  taking lower 8-bits */

                    QUANTIZATION;
					
					out_slm_2[ slm_addr ]   = slm_write0;

				} // out_block_width-for loop
			
			}  // out_block_height-for loop

			//  Wait till all threads in WG finish placing the output
			  barrier(CLK_LOCAL_MEM_FENCE);
			  
			/******************* Read from SLM & Write to Global *************************************/	
			
		    /* Each lane will read uint4 from SLM - 4K x 4N values. Swizzle them into 4N x 4K order

     		   SLM Read Distribution - 8Px14Qx4Nx32K output tile
			
			   Threads 0-1 handles row0, col 0-13,
			   Threads 2-3 handles row1, col 0-13,
			   ..
			   Threads 14-15 handles row7, col 0-13 */

			uint row_id =   thread_id / 2;
			uint col_id =   ( thread_id % 2 )*7;
			
			uint out_slm_read =  col_id * 32 + row_id * TILE_W * 32 + lane_id * 4;
			
			__local uint4 *out_slm3   = (__local uint4*) &out_slm[ out_slm_read ];
			
			/* feature maps are an array of slicePacks, each H,W position within the slice pack contains 32 8bit feature maps(channels) of 8 different batches */
			uint row_size_bytes        = (_OW + OWPAD) * PACK * BATCH_PACK;
			
			/* slice_pack is a pack of 32 feature map tiles that are [OH][OW][4][32] that are stored within the full [K/32][N/4][OH][OW][4][32] output */
			uint slice_pack_size_bytes = row_size_bytes * (_OH + OHPAD); 
			
			/* Each fmg writes [OH][OW][4][32]*/
		
		 	uint output_depth_index      =  output_depth;

			uint batch_index			 =  batch;
			
			uint slice_pack_addr_bytes  = output_depth_index * slice_pack_size_bytes * ( BATCH_SIZE / BATCH_PACK ) + batch_index * slice_pack_size_bytes + (groupz_tile + row_id ) * row_size_bytes + (groupy_tile + col_id ) * PACK * BATCH_PACK; 
			
			__global uint* output_write = (__global uint *) &outputs [ slice_pack_addr_bytes ];
			
			/* Each lane writes 4K values of 4 batches and 8 different columns */
			
			/* 4K values of K=0..31 */
			
			const char  mask_constant = 0xFF;
            
			__attribute__((opencl_unroll_hint(7)))
			for ( int c=0; c<7; c++ )
            {
				/* Get 4K4N values in uint4 - each uint containing 4N values of a K
 				   swizzle the data and pack into another uint4 containing 4N4K values - each uint containing 4K values of a N. 
				   Use block_writes for writing uint4 */
                
				uint4 out_k4n4 = out_slm3 [ c*8 ];

               	//Pack 4K values of first n
				uchar4 out_n0k4;

				out_n0k4.s0 = out_k4n4.s0 & mask_constant;
				out_n0k4.s1 = out_k4n4.s1 & mask_constant;
				out_n0k4.s2 = out_k4n4.s2 & mask_constant;
				out_n0k4.s3 = out_k4n4.s3 & mask_constant;
		
		        /* Assigning to uchar hence need to get the required bits to lower 8-bits*/
				
				//Pack 4K values of second n		
				uchar4 out_n1k4;
				
			    out_n1k4.s0 = (out_k4n4.s0 >> 8) & mask_constant;
				out_n1k4.s1 = (out_k4n4.s1 >> 8) & mask_constant;
				out_n1k4.s2 = (out_k4n4.s2 >> 8) & mask_constant;
				out_n1k4.s3 = (out_k4n4.s3 >> 8) & mask_constant;

		        //Pack 4K values of third n			
				uchar4 out_n2k4;
				
				out_n2k4.s0  = (out_k4n4.s0 >> 16) & mask_constant;
				out_n2k4.s1  = (out_k4n4.s1 >> 16) & mask_constant;
				out_n2k4.s2  = (out_k4n4.s2 >> 16) & mask_constant;
				out_n2k4.s3  = (out_k4n4.s3 >> 16) & mask_constant;

		        //Pack 4K values of fourth n
				uchar4 out_n3k4;

				out_n3k4.s0 = (out_k4n4.s0 >> 24) & mask_constant;
				out_n3k4.s1 = (out_k4n4.s1 >> 24) & mask_constant;
				out_n3k4.s2 = (out_k4n4.s2 >> 24) & mask_constant;
				out_n3k4.s3 = (out_k4n4.s3 >> 24) & mask_constant;
				
				uint4 out_n4k4;
				
				out_n4k4.s0 = as_uint ( out_n0k4 );
				out_n4k4.s1 = as_uint ( out_n1k4 );
				out_n4k4.s2 = as_uint ( out_n2k4 );
				out_n4k4.s3 = as_uint ( out_n3k4 );
								
			    intel_sub_group_block_write4 ( output_write , out_n4k4 );

				output_write += 4*8;
			}
} //end of kernel

#undef SCAL
#undef QUANTIZATION
