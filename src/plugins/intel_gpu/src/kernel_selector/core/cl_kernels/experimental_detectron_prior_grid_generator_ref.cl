// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

KERNEL (experimental_detectron_prior_grid_generator_ref)(const __global INPUT0_TYPE *input, __global OUTPUT_TYPE *output)
{
    const size_t y = get_global_id(0);
    const size_t x = get_global_id(1);
    const size_t prior = get_global_id(2);
    const ptrdiff_t inputOffset = INPUT0_OFFSET + INPUT0_PITCHES[3] * prior; // input shape is [ 1, 1, 4, number_of_priors]
    const float stepX = STEP_X * ( x + 0.5f );
    const float stepY = STEP_Y * ( y + 0.5f );
    const float roi0 = input[inputOffset] + stepX;
    const float roi1 = input[inputOffset + INPUT0_PITCHES[2]] + stepY;
    const float roi2 = input[inputOffset + INPUT0_PITCHES[2] * 2] + stepX;
    const float roi3 = input[inputOffset + INPUT0_PITCHES[2] * 3] + stepY;
#ifdef NUM_PRIORS // output shape is [4, featmap_height * featmap_width * number_of_priors, 1, 1], some elements may be left unset
    const ptrdiff_t outputOffset = OUTPUT_OFFSET + OUTPUT_PITCHES[1] * ( LAYER_WIDTH * NUM_PRIORS * y + NUM_PRIORS * x + prior);
#else // output shape is [4, number_of_priors, featmap_width, featmap_height], some elements may be left unset
    const ptrdiff_t outputOffset = OUTPUT_OFFSET + OUTPUT_PITCHES[3] * y + OUTPUT_PITCHES[2] * x + OUTPUT_PITCHES[1] * prior;
#endif
    output[outputOffset] = roi0;
    output[outputOffset + OUTPUT_PITCHES[0]] = roi1;
    output[outputOffset + OUTPUT_PITCHES[0] * 2] = roi2;
    output[outputOffset + OUTPUT_PITCHES[0] * 3] = roi3;
}
