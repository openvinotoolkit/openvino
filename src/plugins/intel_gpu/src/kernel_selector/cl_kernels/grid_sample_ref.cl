// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

KERNEL(grid_sample_ref)
(const __global INPUT0_TYPE* data, const __global INPUT1_TYPE* grid, __global OUTPUT_TYPE* output) {
    const uint bf = get_global_id(0);
    const uint b = bf % OUTPUT_BATCH_NUM;
    const uint f = bf / OUTPUT_BATCH_NUM;
    const uint y = get_global_id(1);
    const uint x = get_global_id(2);

#ifdef ALIGN_CORNERS
    printf("ALIGN_CORNERS = true\n");
#else
    printf("ALIGN_CORNERS = false\n");
#endif

#ifdef INTERPOLATION_MODE_BILINEAR
    printf("INTERPOLATION_MODE_BILINEAR = true\n");
#endif
#ifdef INTERPOLATION_MODE_BICUBIC
    printf("INTERPOLATION_MODE_BICUBIC = true\n");
#endif
#ifdef INTERPOLATION_MODE_NEAREST
    printf("INTERPOLATION_MODE_NEAREST = true\n");
#endif

#ifdef PADDING_MODE_ZEROS
    printf("PADDING_MODE_ZEROS = true\n");
#endif
#ifdef PADDING_MODE_BORDER
    printf("PADDING_MODE_BORDER = true\n");
#endif
#ifdef PADDING_MODE_REFLECTION
    printf("PADDING_MODE_REFLECTION = true\n");
#endif

    output[OUTPUT_GET_INDEX(b, f, y, x)] = data[INPUT0_GET_INDEX(b, f, y, x)];
}
