// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

typedef __global INPUT0_TYPE data_t;
typedef __global INPUT1_TYPE grid_t;
typedef __global OUTPUT_TYPE output_t;

typedef INPUT0_TYPE data_et;
typedef float grid_et;
typedef OUTPUT_TYPE output_et;

#ifdef VOLUMETRIC

// ---------------------------------------------------------------------------
// 5D (volumetric) GridSample. data: [N, C, D, H, W], grid: [N, D_out, H_out, W_out, 3]
// where the last grid dimension stores (x, y, z) with x->W, y->H, z->D.
// ---------------------------------------------------------------------------

inline const data_et FUNC(get_data_single_value)(const data_t* buffer,
                                                  const size_t n,
                                                  const size_t c,
                                                  const size_t z,
                                                  const size_t y,
                                                  const size_t x) {
    const size_t idx = INPUT0_GET_INDEX(n, c, z, y, x);
    return buffer[idx];
}
#define get_data_single_value FUNC_CALL(get_data_single_value)

inline const grid_et FUNC(get_grid_single_value)(const grid_t* buffer,
                                                  const size_t n,
                                                  const size_t coord,
                                                  const size_t z,
                                                  const size_t y,
                                                  const size_t x) {
    const size_t idx = INPUT1_GET_INDEX(n, z, y, x, coord);
    return buffer[idx];
}
#define get_grid_single_value FUNC_CALL(get_grid_single_value)

inline void FUNC(set_output_single_value)(const output_et value,
                                          output_t* buffer,
                                          const size_t n,
                                          const size_t c,
                                          const size_t z,
                                          const size_t y,
                                          const size_t x) {
    const size_t idx = OUTPUT_GET_INDEX(n, c, z, y, x);
    buffer[idx] = value;
}
#define set_output_single_value FUNC_CALL(set_output_single_value)

#if defined(ALIGN_CORNERS)
#define rescale_align FUNC(denormalize)
inline grid_et rescale_align(const grid_et value, const size_t range) {
    return (value + 1) * ((grid_et)(range)-1) / 2;
}
#else
#define rescale_noalign FUNC(denormalize)
inline grid_et rescale_noalign(const grid_et value, const size_t range) {
    return ((value + 1) * (grid_et)(range)-1) / 2;
}
#endif
#define denormalize FUNC_CALL(denormalize)

#if defined(PADDING_MODE_ZEROS)
#define zeros_padding FUNC(get_padded)
inline data_et zeros_padding(const data_t* data,
                             const size_t n,
                             const size_t c,
                             const long z_d,
                             const long y_d,
                             const long x_d) {
    const long D = convert_long(INPUT0_SIZE_Z);
    const long H = convert_long(INPUT0_SIZE_Y);
    const long W = convert_long(INPUT0_SIZE_X);
    if (z_d < 0 || y_d < 0 || x_d < 0 || z_d >= D || y_d >= H || x_d >= W) {
        return 0;
    } else {
        return get_data_single_value(data, n, c, (size_t)(z_d), (size_t)(y_d), (size_t)(x_d));
    }
}
#undef zeros_padding

#elif defined(PADDING_MODE_BORDER)
#define border_padding FUNC(get_padded)
inline data_et border_padding(const data_t* data,
                              const size_t n,
                              const size_t c,
                              const long z_d,
                              const long y_d,
                              const long x_d) {
    const long D = convert_long(INPUT0_SIZE_Z);
    const long H = convert_long(INPUT0_SIZE_Y);
    const long W = convert_long(INPUT0_SIZE_X);
    const size_t z = (size_t)(clamp(z_d, 0l, D - 1));
    const size_t y = (size_t)(clamp(y_d, 0l, H - 1));
    const size_t x = (size_t)(clamp(x_d, 0l, W - 1));
    return get_data_single_value(data, n, c, z, y, x);
}
#undef border_padding

#elif defined(PADDING_MODE_REFLECTION)

#if defined(ALIGN_CORNERS)
inline long FUNC(reflect_axis)(long coord, const long size) {
    const long span = size == 1 ? 1 : 2 * (size - 1);
    coord = abs(coord) % span;
    return coord >= size ? span - coord : coord;
}
#else
inline long FUNC(reflect_axis)(long coord, const long size) {
    const long span = size * 2l;
    coord = (coord % span + span) % span;
    return coord >= size ? span - 1 - coord : coord;
}
#endif
#define reflect_axis FUNC_CALL(reflect_axis)

#define reflection_padding FUNC(get_padded)
inline data_et reflection_padding(const data_t* data,
                                  const size_t n,
                                  const size_t c,
                                  long z_d,
                                  long y_d,
                                  long x_d) {
    const size_t z = (size_t)reflect_axis(z_d, convert_long(INPUT0_SIZE_Z));
    const size_t y = (size_t)reflect_axis(y_d, convert_long(INPUT0_SIZE_Y));
    const size_t x = (size_t)reflect_axis(x_d, convert_long(INPUT0_SIZE_X));
    return get_data_single_value(data, n, c, z, y, x);
}
#undef reflection_padding
#else
#error [clDNN grid_sample_ref.cl]: undefined padding mode
#endif

#define get_padded FUNC_CALL(get_padded)

#if defined(INTERPOLATION_MODE_BILINEAR)
#define trilinear FUNC(interpolate)
inline data_et trilinear(const data_t* data,
                         const size_t n,
                         const size_t c,
                         const grid_et z_n,
                         const grid_et y_n,
                         const grid_et x_n) {
    const grid_et z_d = denormalize(z_n, INPUT0_SIZE_Z);
    const grid_et y_d = denormalize(y_n, INPUT0_SIZE_Y);
    const grid_et x_d = denormalize(x_n, INPUT0_SIZE_X);
    const grid_et z0 = floor(z_d);
    const grid_et y0 = floor(y_d);
    const grid_et x0 = floor(x_d);
    const grid_et dz = z_d - z0;
    const grid_et dy = y_d - y0;
    const grid_et dx = x_d - x0;

    const data_et v000 = get_padded(data, n, c, z0, y0, x0);
    const data_et v001 = get_padded(data, n, c, z0, y0, x0 + 1);
    const data_et v010 = get_padded(data, n, c, z0, y0 + 1, x0);
    const data_et v011 = get_padded(data, n, c, z0, y0 + 1, x0 + 1);
    const data_et v100 = get_padded(data, n, c, z0 + 1, y0, x0);
    const data_et v101 = get_padded(data, n, c, z0 + 1, y0, x0 + 1);
    const data_et v110 = get_padded(data, n, c, z0 + 1, y0 + 1, x0);
    const data_et v111 = get_padded(data, n, c, z0 + 1, y0 + 1, x0 + 1);

    const data_et c00 = (1 - dx) * v000 + dx * v001;
    const data_et c01 = (1 - dx) * v010 + dx * v011;
    const data_et c10 = (1 - dx) * v100 + dx * v101;
    const data_et c11 = (1 - dx) * v110 + dx * v111;

    const data_et c0 = (1 - dy) * c00 + dy * c01;
    const data_et c1 = (1 - dy) * c10 + dy * c11;

    return (1 - dz) * c0 + dz * c1;
}
#undef trilinear

#elif defined(INTERPOLATION_MODE_NEAREST)
#define nearest3d FUNC(interpolate)
inline data_et nearest3d(const data_t* data,
                         const size_t n,
                         const size_t c,
                         const grid_et z_n,
                         const grid_et y_n,
                         const grid_et x_n) {
    const grid_et z_nearest = rint(denormalize(z_n, INPUT0_SIZE_Z));
    const grid_et y_nearest = rint(denormalize(y_n, INPUT0_SIZE_Y));
    const grid_et x_nearest = rint(denormalize(x_n, INPUT0_SIZE_X));
    return get_padded(data, n, c, z_nearest, y_nearest, x_nearest);
}
#undef nearest3d
#else
#error [clDNN grid_sample_ref.cl]: unsupported interpolation mode for volumetric GridSample
#endif

#define interpolate FUNC_CALL(interpolate)

KERNEL(grid_sample_ref)(const data_t * data,
                        const grid_t * grid,
                        output_t * output
                        #if HAS_FUSED_OPS_DECLS
                        , FUSED_OPS_DECLS
                        #endif
                       ) {
#if INPUT0_BATCH_NUM != INPUT1_BATCH_NUM
#error [clDNN grid_sample_ref.cl]: the batch dimension in the input data tensor's shape doesn't match the batch dimension in the grid tensor's shape
#endif

#if INPUT1_SIZE_X != 3
#error [clDNN grid_sample_ref.cl]: the last dimension of the grid tensor must be 3 for volumetric GridSample
#endif

    const uint nc = get_global_id(0);
    const uint n = nc % OUTPUT_BATCH_NUM;
    const uint c = nc / OUTPUT_BATCH_NUM;
    const uint zy = get_global_id(1);
    const uint z = zy / OUTPUT_SIZE_Y;
    const uint y = zy % OUTPUT_SIZE_Y;
    const uint x = get_global_id(2);

    const grid_et x_n = get_grid_single_value(grid, n, 0, z, y, x);
    const grid_et y_n = get_grid_single_value(grid, n, 1, z, y, x);
    const grid_et z_n = get_grid_single_value(grid, n, 2, z, y, x);

    const output_et out = interpolate(data, n, c, z_n, y_n, x_n);

    set_output_single_value(out, output, n, c, z, y, x);
}
#undef interpolate
#undef get_padded
#undef denormalize

#else  // !VOLUMETRIC  (original 4D path - unchanged)

inline const data_et FUNC(
    get_data_single_value)(const data_t* buffer, const size_t n, const size_t c, const size_t h, const size_t w) {
    const size_t idx = INPUT0_GET_INDEX(n, c, h, w);
    return buffer[idx];
}
#define get_data_single_value FUNC_CALL(get_data_single_value)

inline const grid_et FUNC(
    get_grid_single_value)(const grid_t* buffer, const size_t n, const size_t c, const size_t h, const size_t w) {
    const size_t idx = INPUT1_GET_INDEX(n, h, w, c);
    return buffer[idx];
}
#define get_grid_single_value FUNC_CALL(get_grid_single_value)

inline void FUNC(set_output_single_value)(const output_et value,
                                          output_t* buffer,
                                          const size_t n,
                                          const size_t c,
                                          const size_t h,
                                          const size_t w) {
    const size_t idx = OUTPUT_GET_INDEX(n, c, h, w);
    buffer[idx] = value;
}
#define set_output_single_value FUNC_CALL(set_output_single_value)

#if defined(ALIGN_CORNERS)
#define rescale_align FUNC(denormalize)
inline grid_et rescale_align(const grid_et value, const size_t range) {
    return (value + 1) * ((grid_et)(range)-1) / 2;
}
#else
#define rescale_noalign FUNC(denormalize)
inline grid_et rescale_noalign(const grid_et value, const size_t range) {
    return ((value + 1) * (grid_et)(range)-1) / 2;
}
#endif
#define denormalize FUNC_CALL(denormalize)

#if defined(PADDING_MODE_ZEROS)
#define zeros_padding FUNC(get_padded)
inline data_et zeros_padding(const data_t* data, const size_t n, const size_t c, const long y_d, const long x_d) {
    const long H = convert_long(INPUT0_SIZE_Y);
    const long W = convert_long(INPUT0_SIZE_X);
    if (y_d < 0 || x_d < 0 || y_d >= H || x_d >= W) {
        return 0;
    } else {
        const size_t y = (size_t)(y_d);
        const size_t x = (size_t)(x_d);
        return get_data_single_value(data, n, c, y, x);
    }
}
#undef zeros_padding

#elif defined(PADDING_MODE_BORDER)
#define border_padding FUNC(get_padded)
inline data_et border_padding(const data_t* data, const size_t n, const size_t c, const long y_d, const long x_d) {
    const long H = INPUT0_SIZE_Y;
    const long W = INPUT0_SIZE_X;
    const size_t y = (size_t)(clamp(y_d, 0l, H - 1));
    const size_t x = (size_t)(clamp(x_d, 0l, W - 1));
    return get_data_single_value(data, n, c, y, x);
}
#undef border_padding

#elif defined(PADDING_MODE_REFLECTION)

#if defined(ALIGN_CORNERS)
#define reflection_data_with_align FUNC(get_padded)
inline data_et reflection_data_with_align(const data_t* data, const size_t n, const size_t c, long y_d, long x_d) {
    const long H = convert_long(INPUT0_SIZE_Y);
    const long W = convert_long(INPUT0_SIZE_X);
    const long H_2_2 = H == 1 ? 1 : 2 * (H - 1);
    const long W_2_2 = W == 1 ? 1 : 2 * (W - 1);
    y_d = abs(y_d) % H_2_2;
    x_d = abs(x_d) % W_2_2;
    const size_t y = (size_t)(y_d >= H ? H_2_2 - y_d : y_d);
    const size_t x = (size_t)(x_d >= W ? W_2_2 - x_d : x_d);
    return get_data_single_value(data, n, c, y, x);
}
#undef reflection_data_with_align

#else
#define reflection_data_no_align FUNC(get_padded)
inline data_et reflection_data_no_align(const data_t* data, const size_t n, const size_t c, long y_d, long x_d) {
    const long H = convert_long(INPUT0_SIZE_Y);
    const long W = convert_long(INPUT0_SIZE_X);
    const long H_2 = convert_long(INPUT0_SIZE_Y) * 2l;
    const long W_2 = convert_long(INPUT0_SIZE_X) * 2l;
    y_d = (y_d % H_2 + H_2) % H_2;
    x_d = (x_d % W_2 + W_2) % W_2;
    const size_t y = (size_t)(y_d >= H ? H_2 - 1 - y_d : y_d);
    const size_t x = (size_t)(x_d >= W ? W_2 - 1 - x_d : x_d);
    return get_data_single_value(data, n, c, y, x);
}
#undef reflection_data_no_align
#endif
#else
#error [clDNN grid_sample_ref.cl]: undefined padding mode
#endif

#define get_padded FUNC_CALL(get_padded)

#if defined(INTERPOLATION_MODE_BILINEAR)
#define bilinear FUNC(interpolate)
inline data_et bilinear(const data_t* data, const size_t n, const size_t c, const grid_et y_n, const grid_et x_n) {
    const grid_et y_d = denormalize(y_n, INPUT0_SIZE_Y);
    const grid_et x_d = denormalize(x_n, INPUT0_SIZE_X);
    const grid_et y_topleft = floor(y_d);
    const grid_et x_topleft = floor(x_d);
    const grid_et dy = y_d - y_topleft;
    const grid_et dx = x_d - x_topleft;
    const data_et v00 = get_padded(data, n, c, y_topleft, x_topleft);
    const data_et v01 = get_padded(data, n, c, y_topleft, x_topleft + 1);
    const data_et v10 = get_padded(data, n, c, y_topleft + 1, x_topleft);
    const data_et v11 = get_padded(data, n, c, y_topleft + 1, x_topleft + 1);

    const data_et q0 = (1 - dx) * v00 + dx * v01;
    const data_et q1 = (1 - dx) * v10 + dx * v11;
    return dy * q1 + (1 - dy) * q0;
}
#undef bilinear

#elif defined(INTERPOLATION_MODE_NEAREST)
#define nearest FUNC(interpolate)
inline data_et nearest(const data_t* data, const size_t n, const size_t c, const grid_et y_n, const grid_et x_n) {
    const grid_et y_nearest = rint(denormalize(y_n, INPUT0_SIZE_Y));
    const grid_et x_nearest = rint(denormalize(x_n, INPUT0_SIZE_X));
    return get_padded(data, n, c, y_nearest, x_nearest);
}
#undef nearest

#elif defined(INTERPOLATION_MODE_BICUBIC)

typedef MAKE_VECTOR_TYPE(INPUT1_TYPE, 4) vector_grid_4_t;
typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) vector_data_4_t;
typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, 16) matrix_data_4x4_t;

inline vector_grid_4_t FUNC(cubic_coeffs)(const data_et r, const data_et A) {
    vector_grid_4_t v;
    v[0] = ((A * (r + 1) - 5 * A) * (r + 1) + 8 * A) * (r + 1) - 4 * A;
    v[1] = ((A + 2) * r - (A + 3)) * r * r + 1;
    v[2] = ((A + 2) * (1 - r) - (A + 3)) * (1 - r) * (1 - r) + 1;
    v[3] = ((A * (2 - r) - 5 * A) * (2 - r) + 8 * A) * (2 - r) - 4 * A;
    return v;
}
#define cubic_coeffs FUNC_CALL(cubic_coeffs)

inline matrix_data_4x4_t FUNC(
    gather_4x4)(const data_t* data, const size_t n, const size_t c, const long y_topleft, const long x_topleft) {
    matrix_data_4x4_t s;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            s[j * 4 + i] = get_padded(data, n, c, y_topleft + j, x_topleft + i);
        }
    }
    return s;
}
#define gather_4x4 FUNC_CALL(gather_4x4)

inline data_et FUNC(inner_product_v4_v4)(const vector_grid_4_t v1, const vector_data_4_t v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}
#define inner_product_v4_v4 FUNC_CALL(inner_product_v4_v4)

inline vector_data_4_t FUNC(inner_product_m4_v4)(const matrix_data_4x4_t m, const vector_grid_4_t v) {
    vector_data_4_t p = {
        m[0 * 4 + 0] * v[0] + m[0 * 4 + 1] * v[1] + m[0 * 4 + 2] * v[2] + m[0 * 4 + 3] * v[3],
        m[1 * 4 + 0] * v[0] + m[1 * 4 + 1] * v[1] + m[1 * 4 + 2] * v[2] + m[1 * 4 + 3] * v[3],
        m[2 * 4 + 0] * v[0] + m[2 * 4 + 1] * v[1] + m[2 * 4 + 2] * v[2] + m[2 * 4 + 3] * v[3],
        m[3 * 4 + 0] * v[0] + m[3 * 4 + 1] * v[1] + m[3 * 4 + 2] * v[2] + m[3 * 4 + 3] * v[3],
    };

    return p;
}
#define inner_product_m4_v4 FUNC_CALL(inner_product_m4_v4)

#define bicubic FUNC(interpolate)
inline data_et bicubic(const data_t* data, const size_t n, const size_t c, const grid_et y_n, const grid_et x_n) {
    const grid_et y_d = denormalize(y_n, INPUT0_SIZE_Y);
    const grid_et x_d = denormalize(x_n, INPUT0_SIZE_X);
    const grid_et y_topleft = floor(y_d);
    const grid_et x_topleft = floor(x_d);
    const grid_et dy = y_d - y_topleft;
    const grid_et dx = x_d - x_topleft;
    matrix_data_4x4_t s = gather_4x4(data, n, c, y_topleft - 1, x_topleft - 1);

    vector_grid_4_t cy = cubic_coeffs(dy, -0.75);
    vector_grid_4_t cx = cubic_coeffs(dx, -0.75);
    vector_data_4_t p;
    p = inner_product_m4_v4(s, cx);
    return inner_product_v4_v4(cy, p);
}
#undef bicubic
#undef inner_product_m4_v4
#undef inner_product_v4_v4
#undef cubic_coeffs
#undef gather_4x4
#else
#error[clDNN grid_sample_ref.cl]: undefined interpolation mode
#endif

#define interpolate FUNC_CALL(interpolate)

KERNEL(grid_sample_ref)(const data_t * data,
                        const grid_t * grid,
                        output_t * output
                        #if HAS_FUSED_OPS_DECLS
                        , FUSED_OPS_DECLS
                        #endif
                       ) {
#if INPUT0_BATCH_NUM != INPUT1_BATCH_NUM
#error [clDNN grid_sample_ref.cl]: the batch dimension in the input data tensor's shape doesn't match the batch dimension in the grid tensor's shape
#endif

#if INPUT1_SIZE_X != 2
#error [clDNN grid_sample_ref.cl]: wrong dimension of grid tensor's
#endif

    const uint nc = get_global_id(0);
    const uint n = nc % OUTPUT_BATCH_NUM;
    const uint c = nc / OUTPUT_BATCH_NUM;
    const uint h = get_global_id(1);
    const uint w = get_global_id(2);

    const grid_et y_n = get_grid_single_value(grid, n, 1, h, w);
    const grid_et x_n = get_grid_single_value(grid, n, 0, h, w);

    const output_et out = interpolate(data, n, c, y_n, x_n);

    set_output_single_value(out, output, n, c, h, w);
}
#undef interpolate
#undef get_padded
#undef denormalize

#endif  // VOLUMETRIC
