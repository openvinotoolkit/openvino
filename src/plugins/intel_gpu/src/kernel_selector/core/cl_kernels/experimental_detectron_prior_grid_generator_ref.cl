// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if OUTPUT_TYPE_SIZE == 2 //f16
#define HALF_ONE 0.5h
#else
#define HALF_ONE 0.5f
#endif

KERNEL (experimental_detectron_prior_grid_generator_ref)(const __global INPUT0_TYPE *input, __global OUTPUT_TYPE *output)
{
    const size_t y = get_global_id(0);
    const size_t x = get_global_id(1);
    const size_t prior = get_global_id(2);
    const ptrdiff_t inputOffset = INPUT0_GET_INDEX(prior, 0, 0, 0);
    const OUTPUT_TYPE stepX = STEP_X * ( x + HALF_ONE );
    const OUTPUT_TYPE stepY = STEP_Y * ( y + HALF_ONE );
#ifdef FLATTEN
    const ptrdiff_t outputOffset = OUTPUT_GET_INDEX(LAYER_WIDTH * INPUT0_BATCH_NUM * y + INPUT0_BATCH_NUM * x + prior, 0, 0, 0);
#else
    const ptrdiff_t outputOffset = OUTPUT_OFFSET + OUTPUT_PITCHES[1] * ( LAYER_WIDTH * INPUT0_BATCH_NUM * y + INPUT0_BATCH_NUM * x + prior);
#endif
    output[outputOffset] = input[inputOffset] + stepX;
    output[outputOffset + 1] =  input[inputOffset + 1] + stepY;
    output[outputOffset + 2] = input[inputOffset + 2] + stepX;
    output[outputOffset + 3] = input[inputOffset + 3] + stepY;
}

#undef HALF_ONE
