// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

typedef INPUT0_TYPE data_t;
typedef INPUT1_TYPE grid_t;
typedef OUTPUT_TYPE output_t;

typedef INPUT0_TYPE data_et;
typedef float grid_et;
typedef OUTPUT_TYPE output_et;

#if defined(ALIGN_CORNERS)
#    define rescale_align FUNC(denormalize)
inline grid_et rescale_align(const grid_et value, const size_t range) {
    return (value + 1) * ((grid_et)(range)-1) / 2;
}
#else
#    define rescale_noalign FUNC(denormalize)
inline grid_et rescale_noalign(const grid_et value, const size_t range) {
    return ((value + 1) * (grid_et)(range)-1) / 2;
}
#endif
#define denormalize FUNC_CALL(denormalize)

inline const bool FUNC(is_between)(int val, int min, int max) {
    return (val >= min) && (val < max);
}
#define is_between FUNC_CALL(is_between)

// ====================================================================
//
// GRID SAMPLE KERNEL (layout-agnostic data/output addressing)
//
// Same batched grid-coordinate-caching strategy as the original bfyx-only
// kernel (decode each spatial location's sampling coordinates once, reuse
// across all channels -- this is the actual performance win over the
// reference kernel, which redoes this math redundantly per (n,c,h,w)
// thread). The only change is that `data`/`output` element addresses now go
// through INPUT0_GET_INDEX/OUTPUT_GET_INDEX -- the kernel_selector's
// standard layout-aware indexing macros (same ones grid_sample_ref.cl uses)
// -- instead of the previous hand-rolled `base + c * H * W` planar-only
// arithmetic, so this kernel works correctly for any layout the tensor
// actually has (bfyx, b_fs_yx_fsv16, ...), not just bfyx. `grid` (input1)
// keeps the original flat block-cached access pattern unchanged: it's a
// [N,H,W,2] auxiliary tensor, not a feature-map subject to the same
// blocked-layout optimization decisions as `data`, so this loop's core
// optimization (reading a whole block of grid values as one contiguous
// stream) is unaffected either way.
//
// ====================================================================

KERNEL(grid_sample_opt_bilinear_zeros)(const __global data_t* restrict data,
                                       const __global grid_t* restrict grid,
                                       __global output_t* restrict output) {
#if !defined(INTERPOLATION_MODE_BILINEAR)
#    error[clDNN grid_sample_opt_bilinear.cl]: This kernel only support bilinear interppolation mode.
#endif

#if !defined(PADDING_MODE_ZEROS)
#    error[clDNN grid_sample_opt_bilinear.cl]: This kernel only support zeros padding mode.
#endif

    const int n = get_global_id(0);

    const int OUTPUT_C_STRIDE = OUTPUT_SIZE_Y * OUTPUT_SIZE_X;
    const int LOCAL_GRID_OFFSET_FOR_THIS_BLOCK = GRID_ITEMS_PER_BLOCK * 2 * get_group_id(1);
    const int BLOCK_SIZE = get_local_size(1);
    const int GRID_ITEMS_FOR_THIS_BLOCK =
        min(OUTPUT_C_STRIDE * 2 - LOCAL_GRID_OFFSET_FOR_THIS_BLOCK, GRID_ITEMS_PER_BLOCK * 2);

    for (int thisThreadHW = get_local_linear_id() * 2; thisThreadHW < GRID_ITEMS_FOR_THIS_BLOCK;
         thisThreadHW += 2 * BLOCK_SIZE) {
        const int globalThisThreadHW = (thisThreadHW + LOCAL_GRID_OFFSET_FOR_THIS_BLOCK) / 2;
        const int h = globalThisThreadHW / OUTPUT_SIZE_X;
        const int w = globalThisThreadHW % OUTPUT_SIZE_X;

        // Layout-aware grid read (was: raw contiguous-block pointer arithmetic, which assumed
        // `grid` is always a simple planar [N,H,W,2] buffer). That assumption broke once `data`
        // (and therefore, coupled through the layout optimizer, `grid`) could be laid out in a
        // blocked format like b_fs_yx_fsv16 -- silently corrupting these reads. INPUT1_GET_INDEX
        // is layout-aware regardless of what `grid`'s actual runtime layout is. Axis mapping
        // (n, h, w, c) -- not (n, c, h, w) -- matches grid_sample_ref.cl's own established
        // convention for this tensor's real [N,H,W,2] shape (OpenVINO's generic 4D tensor
        // descriptor maps dims positionally to batch/feature/y/x regardless of what they
        // semantically represent, so dim1=H is "feature", dim2=W is "y", dim3=2 is "x" here).
        // This does trade away the batched block-cached grid read for a per-thread lookup, but
        // only for these 2 reads -- the channel loop below (the actual hot path) is unaffected.
        const grid_et x_n = grid[INPUT1_GET_INDEX(n, h, w, 0)];
        const grid_et y_n = grid[INPUT1_GET_INDEX(n, h, w, 1)];

        const grid_et y_d = denormalize(y_n, INPUT0_SIZE_Y);
        const grid_et x_d = denormalize(x_n, INPUT0_SIZE_X);
        const int y_topleft = (int)floor(y_d);
        const int x_topleft = (int)floor(x_d);
        const grid_et dy = y_d - y_topleft;
        const grid_et dx = x_d - x_topleft;

        const bool y0_valid = is_between(y_topleft, 0, INPUT0_SIZE_Y);
        const bool y1_valid = is_between(y_topleft + 1, 0, INPUT0_SIZE_Y);
        const bool x0_valid = is_between(x_topleft, 0, INPUT0_SIZE_X);
        const bool x1_valid = is_between(x_topleft + 1, 0, INPUT0_SIZE_X);

        const bool v00_valid = y0_valid && x0_valid;
        const bool v01_valid = y0_valid && x1_valid;
        const bool v10_valid = y1_valid && x0_valid;
        const bool v11_valid = y1_valid && x1_valid;

        // Same "always load, mask afterward" trick as the original kernel (see its
        // comment on avoiding warp-divergence from conditional loads): substitute a
        // safe in-bounds fallback position (0,0) when the true position would be
        // out-of-bounds, and zero the contribution out in the blend below instead of
        // branching around the load itself.
        const int y0c = y0_valid ? y_topleft : 0;
        const int y1c = y1_valid ? (y_topleft + 1) : 0;
        const int x0c = x0_valid ? x_topleft : 0;
        const int x1c = x1_valid ? (x_topleft + 1) : 0;

#pragma unroll
        for (int c = 0; c < OUTPUT_FEATURE_NUM; ++c) {
            const data_et v00_d = data[INPUT0_GET_INDEX(n, c, y0c, x0c)];
            const data_et v01_d = data[INPUT0_GET_INDEX(n, c, y0c, x1c)];
            const data_et v10_d = data[INPUT0_GET_INDEX(n, c, y1c, x0c)];
            const data_et v11_d = data[INPUT0_GET_INDEX(n, c, y1c, x1c)];

            const data_et v00 = v00_valid ? v00_d * (1 - dx) : 0;
            const data_et v01 = v01_valid ? v01_d * dx : 0;
            const data_et v10 = v10_valid ? v10_d * (1 - dx) : 0;
            const data_et v11 = v11_valid ? v11_d * dx : 0;

            const data_et q0 = v00 + v01;
            const data_et q1 = v10 + v11;
            const data_et out = dy * q1 + (1 - dy) * q0;

            output[OUTPUT_GET_INDEX(n, c, h, w)] = out;
        }
    }
}

#undef denormalize
#undef is_between
