// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "include/batch_headers/fetch_data.cl"
#define ACCUMULATOR_VEC MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, OUTPUT_DIM)
#define FINAL_ACCUMULATOR_VEC MAKE_VECTOR_TYPE(FINAL_ACCUMULATOR_TYPE, OUTPUT_DIM)
#define OUTPUT_VEC MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_DIM)

KERNEL(reduce_simple)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    __local ACCUMULATOR_TYPE buffer[NUM_BLOCKS][OUTPUT_DIM];

    const uint bi = (uint)get_global_id(0);

    ACCUMULATOR_TYPE acc[OUTPUT_DIM];
	#pragma unroll
	for (uint j = 0; j < OUTPUT_DIM; j++) {
		acc[j] = data[bi + j * TOTAL_NUM_ELEMENTS/OUTPUT_DIM];
	}
	
	for (uint i = bi + BLOCK_STRIDE; i < TOTAL_NUM_ELEMENTS/OUTPUT_DIM; i += BLOCK_STRIDE) {
		#pragma unroll
		for (uint j = 0; j < OUTPUT_DIM; j++) {
			#ifdef REDUCE_SUM_MODE
					acc[j] += data[i + j * TOTAL_NUM_ELEMENTS/OUTPUT_DIM];
			#elif REDUCE_MEAN_MODE
					acc[j] += data[i + j * TOTAL_NUM_ELEMENTS/OUTPUT_DIM];
			#elif REDUCE_MAX_MODE
					acc[j] = data[i + j * TOTAL_NUM_ELEMENTS/OUTPUT_DIM] > acc[j] ? data[i + j * TOTAL_NUM_ELEMENTS/OUTPUT_DIM] : acc[j];
			#elif REDUCE_MIN_MODE
					acc[j] = data[i] < acc[j] ? data[i + j * TOTAL_NUM_ELEMENTS/OUTPUT_DIM] : acc[j];
			#elif REDUCE_PROD_MODE
					acc[j] *= data[i + j * TOTAL_NUM_ELEMENTS/OUTPUT_DIM];
			#endif
		}
	}
	
	#pragma unroll
	for (uint j = 0; j < OUTPUT_DIM; j++) {
		buffer[bi][j] = acc[j];
	}			
	
	barrier(CLK_LOCAL_MEM_FENCE);
		
	if (bi != 0)
		return;
	
	#pragma unroll
	for (uint j = 0; j < OUTPUT_DIM; j++) {	
		acc[j] = buffer[0][j];
		
		#pragma unroll 16
		for (uint i = 1; i < NUM_BLOCKS; i++) {
			#ifdef REDUCE_SUM_MODE
				acc[j] += buffer[i][j];
			#elif REDUCE_MEAN_MODE
				acc[j] += buffer[i][j];
			#elif REDUCE_MAX_MODE
				acc[j] = buffer[i][j] > acc[j] ? buffer[i][j] : acc[j];
			#elif REDUCE_MIN_MODE
				acc[j] = buffer[i][j] < acc[j] ? buffer[i][j] : acc[j];
			#elif REDUCE_PROD_MODE
				acc[j] *= buffer[i][j];
			#endif
		}
	}
	
	#pragma unroll
	for (uint j = 0; j < OUTPUT_DIM; j++) {
		FINAL_ACCUMULATOR_TYPE final_acc = TO_FINAL_ACCUMULATOR_TYPE(acc[j]);
		#if REDUCE_MEAN_MODE
			final_acc /= (TOTAL_NUM_ELEMENTS/OUTPUT_DIM);
		#endif
		
		OUTPUT_TYPE final_result;
		ACTIVATION_TYPE reduce_result = TO_ACTIVATION_TYPE(final_acc);
		#if HAS_FUSED_OPS
			FUSED_OPS;
			final_result = FUSED_OPS_RESULT;
		#else
			final_result = TO_OUTPUT_TYPE(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
		#endif
		output[j] = final_result;
	}
}
