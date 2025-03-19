// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#ifdef RTE_OUTPUT
    #define TO_OUTPUT_TYPE(x)   CAT(CAT(convert_, OUTPUT_TYPE), _rte)(x)
#endif

inline int FUNC(get_nearest_val)(float num, bool is_downsample)
{
#if defined(NEAREST_ROUND_PREFER_FLOOR)
    return (num == (int)num + 0.5f) ? (int)floor(num) : (int)round(num);
#elif defined(NEAREST_ROUND_PREFER_CEIL)
    return (int)round(num);
#elif defined(NEAREST_FLOOR)
    return (int)floor(num);
#elif defined(NEAREST_CEIL)
    return (int)ceil(num);
#elif defined(NEAREST_SIMPLE)
    return is_downsample ? (int)ceil(num) : (int)num;
#else
#error [clDNN resample_ref.cl]: nearest mode - not supported
#endif
}

inline float FUNC(get_original_coordinate)(float num, float scale, int length_resized, int length_original)
{
    if (scale == 1.0f)
        return num;
#if defined(COORD_TRANS_MODE_HALF_PIXEL)
    return (num + 0.5f) * scale - 0.5f;
#elif defined(COORD_TRANS_MODE_PYTORCH_HALF_PIXEL)
    return (length_resized > 1) ? (num + 0.5f) * scale - 0.5f : 0.f;
#elif defined(COORD_TRANS_MODE_ASYMMETRIC)
    return num * scale;
#elif defined(COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN)
    return (num + 0.5f) * scale;
#elif defined(COORD_TRANS_MODE_ALIGN_CORNERS)
    return (length_resized != 1) ? num * (length_original - 1) / (length_resized - 1) : 0.f;
#else
#error [clDNN resample_ref.cl]: coordinate transformation mode - not supported
#endif
}

inline void FUNC(get_cubic_coeff)(float* cubic_coef, float coord, float coef)
{
    float abs_num = fabs(coord);
    cubic_coef[0] = coef * (abs_num - 1.0) * (abs_num - 1.0) * abs_num;
    cubic_coef[1] = ((coef + 2.0) * abs_num - (coef + 3.0)) * abs_num * abs_num + 1.0;
    cubic_coef[2] = (((-coef - 2.0) * abs_num + (2.0 * coef + 3.0)) * abs_num - coef) * abs_num;
    cubic_coef[3] = -coef * abs_num * abs_num * (abs_num - 1.0);
}

#define TRIANGLE_COEFF(x) (ACCUMULATOR_MAX_FUNC(ACCUMULATOR_VAL_ZERO, ACCUMULATOR_VAL_ONE - ACCUMULATOR_ABS_FUNC(x)))

KERNEL (resample_gpu_ref)(__global INPUT0_TYPE* input,
                          __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                          , FUSED_OPS_DECLS
#endif
)
{
    const int in_size[5] = { INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X };
    const int out_size[5] = { OUTPUT_BATCH_NUM, OUTPUT_FEATURE_NUM, OUTPUT_SIZE_Z, OUTPUT_SIZE_Y, OUTPUT_SIZE_X };
#if defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) in_pack_t;
    typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, PACK_SIZE) out_pack_t;

    int out_coords[5];
    out_coords[4] = get_global_id(0);
#if OUTPUT_DIMS <= 4
    out_coords[3] = get_global_id(1);
    out_coords[2] = 0;
#else // OUTPUT_DIMS <= 4
    out_coords[3] = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    out_coords[2] = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif //  OUTPUT_DIMS <= 4
    out_coords[1] = ((int)get_global_id(2) * PACK_SIZE) % OUTPUT_FEATURE_NUM;
    out_coords[0] = ((int)get_global_id(2) * PACK_SIZE) / OUTPUT_FEATURE_NUM;
    int in_coords[5];
    bool isOutOfBounds = false;
    unroll_for (int i = 0; i < 5; ++i) {
        const float orig_coord = FUNC_CALL(get_original_coordinate)(out_coords[i], SCALES[i], out_size[i], in_size[i] + PADS_BEGIN[i] +  PADS_END[i]);
        const int nearest_pixel = FUNC_CALL(get_nearest_val)(orig_coord, SCALES[i] > 1) - PADS_BEGIN[i];
        in_coords[i] = max(-PADS_BEGIN[0], min(nearest_pixel, in_size[i] + PADS_END[i] - 1));
#if PADDING_USED == 1
        if (in_coords[i] < 0 || in_coords[i] >= in_size[i])
            isOutOfBounds = true;
#endif
    }

    uint input_idx = FUNC_CALL(get_input_index)(in_coords[0], in_coords[1], 0, in_coords[2], in_coords[3], in_coords[4]);
    uint output_idx = FUNC_CALL(get_output_index)(out_coords[0], out_coords[1], 0, out_coords[2], out_coords[3], out_coords[4]);

    in_pack_t interp_val_pack = ((const __global in_pack_t*)(input + input_idx))[0];
    out_pack_t res;
    unroll_for (uint pi = 0; pi < PACK_SIZE; ++pi) {
        INPUT0_TYPE interp_val = interp_val_pack[pi];
#if PADDING_USED == 1
        if (isOutOfBounds)
            interp_val = INPUT0_VAL_ZERO;
#endif
    #if HAS_FUSED_OPS
        #define batch (out_coords[0])
        #define OF_ID (out_coords[1] + pi)
        #define oz (out_coords[2])
        #define oy (out_coords[3])
        #define ox (out_coords[4])
        FUSED_OPS;
        res[pi] = FUSED_OPS_RESULT;
        #undef batch
        #undef OF_ID
        #undef oz
        #undef oy
        #undef ox
    #else // HAS_FUSED_OPS
        res[pi] = ACTIVATION(interp_val, ACTIVATION_PARAMS);
    #endif // HAS_FUSED_OPS
    }
    ((__global out_pack_t*)(output + output_idx))[0] = res;

#elif defined(SAMPLE_TYPE_NEAREST) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    int out_coords[5];
    out_coords[4] = get_global_id(0);
#if OUTPUT_DIMS <= 4
    out_coords[3] = get_global_id(1);
    out_coords[2] = 0;
#else // OUTPUT_DIMS <= 4
    out_coords[3] = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    out_coords[2] = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif // OUTPUT_DIMS <= 4
    out_coords[1] = (int)get_global_id(2) % OUTPUT_FEATURE_NUM;
    out_coords[0] = (int)get_global_id(2) / OUTPUT_FEATURE_NUM;
    int in_coords[5];
    bool isOutOfBounds = false;
    unroll_for (int i = 0; i < 5; ++i) {
        const float orig_coord = FUNC_CALL(get_original_coordinate)(out_coords[i], SCALES[i], out_size[i], in_size[i] + PADS_BEGIN[i] + PADS_END[i]);
        int nearest_pixel = FUNC_CALL(get_nearest_val)(orig_coord, SCALES[i] > 1) - PADS_BEGIN[i];
        in_coords[i] = max(-PADS_BEGIN[i], min(nearest_pixel, in_size[i] + PADS_END[i] - 1));
#if PADDING_USED == 1
        if (in_coords[i] < 0 || in_coords[i] >= in_size[i])
            isOutOfBounds = true;
#endif
    }
    INPUT0_TYPE interp_val = input[FUNC_CALL(get_input_index)(in_coords[0], in_coords[1], 0, in_coords[2], in_coords[3], in_coords[4])];
#if PADDING_USED == 1
    if (isOutOfBounds)
        interp_val = INPUT0_VAL_ZERO;
#endif
#if HAS_FUSED_OPS
    #define batch (out_coords[0])
    #define OF_ID (out_coords[1])
    #define oz (out_coords[2])
    #define oy (out_coords[3])
    #define ox (out_coords[4])
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    #undef batch
    #undef OF_ID
    #undef oz
    #undef oy
    #undef ox
#else // HAS_FUSED_OPS
    OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(interp_val), ACTIVATION_PARAMS);
#endif // HAS_FUSED_OPS
    output[FUNC_CALL(get_output_index)(out_coords[0], out_coords[1], 0, out_coords[2], out_coords[3], out_coords[4])] = res;
#elif defined(SAMPLE_TYPE_CUBIC) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    int out_coords[5];
    out_coords[4] = get_global_id(0);
#if OUTPUT_DIMS <= 4
    out_coords[3] = get_global_id(1);
    out_coords[2] = 0;
#else // OUTPUT_DIMS <= 4
    out_coords[3] = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    out_coords[2] = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif // OUTPUT_DIMS <= 4
    out_coords[1] = (int)get_global_id(2) % OUTPUT_FEATURE_NUM;
    out_coords[0] = (int)get_global_id(2) / OUTPUT_FEATURE_NUM;
    int in_coords[5];
    float cubic_coeff[5][4];
    unroll_for (int i = 0; i < 5; ++i) {
        float orig_coord = FUNC_CALL(get_original_coordinate)(out_coords[i], SCALES[i], out_size[i], in_size[i] + PADS_BEGIN[i] + PADS_END[i]) - PADS_BEGIN[i];
        in_coords[i] = floor(orig_coord);
        orig_coord = (orig_coord - in_coords[i]) * AXES_USED[i];
        FUNC_CALL(get_cubic_coeff)(cubic_coeff[i], orig_coord, CUBE_COEFF);
    }

    INPUT0_TYPE interp_val = INPUT0_VAL_ZERO;
    int index[5];
    unroll_for (index[0] = 0; index[0] <= 3; ++index[0]) {
        unroll_for (index[1] = 0; index[1] <= 3; ++index[1]) {
            unroll_for (index[2] = 0; index[2] <= 3; ++index[2]) {
                unroll_for (index[3] = 0; index[3] <= 3; ++index[3]) {
                    unroll_for (index[4] = 0; index[4] <= 3; ++index[4]) {
                        int coords_sum[5] = { in_coords[0], in_coords[1], in_coords[2], in_coords[3], in_coords[4] };
                        float coeff_prod = 1.0f;
                        bool isOutOfBounds = false;
                        unroll_for (int i = 0; i < 5; ++i) {
                            coords_sum[i] = max(-PADS_BEGIN[i], min(in_coords[i] + index[i] - 1, PADS_END[i] + in_size[i] - 1));
#if PADDING_USED == 1
                            if (coords_sum[i] < 0 || coords_sum[i] >= in_size[i])
                                isOutOfBounds = true;
#endif
                            coeff_prod *= cubic_coeff[i][index[i]];
                        }
#if PADDING_USED == 1
                        if (!isOutOfBounds)
#endif
                            interp_val += coeff_prod * input[FUNC_CALL(get_input_index)(coords_sum[0], coords_sum[1], 0, coords_sum[2], coords_sum[3], coords_sum[4])];
                    }
                }
            }
        }
    }

#if HAS_FUSED_OPS
    #define batch (out_coords[0])
    #define OF_ID (out_coords[1])
    #define oz (out_coords[2])
    #define oy (out_coords[3])
    #define ox (out_coords[4])
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    #undef batch
    #undef OF_ID
    #undef oz
    #undef oy
    #undef ox
#else // HAS_FUSED_OPS
    OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(interp_val), ACTIVATION_PARAMS);
#endif // HAS_FUSED_OPS
    output[FUNC_CALL(get_output_index)(out_coords[0], out_coords[1], 0, out_coords[2], out_coords[3], out_coords[4])] = res;
#elif defined(SAMPLE_TYPE_LINEAR_ONNX) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE

#if OUTPUT_DIMS <= 4
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int feature = 0;
    const int batch = get_global_id(2);
    const int PADDED_Y = in_size[3] + PADS_BEGIN[3] + PADS_END[3];
    const int PADDED_X = in_size[4] + PADS_BEGIN[4] + PADS_END[4];
    const float ix = FUNC_CALL(get_original_coordinate)(ox, SCALES[4], out_size[4], PADDED_X);
    const float iy = FUNC_CALL(get_original_coordinate)(oy, SCALES[3], out_size[3], PADDED_Y);

#ifdef LEFTOVERS
    if (ox >= OUTPUT_SIZE_X)
        return;
#endif

    float in_y = fmax(0, fmin(iy, PADDED_Y - 1));
    float in_x = fmax(0, fmin(ix, PADDED_X - 1));
    int in_y1 = min((int)in_y, PADDED_Y - 1);
    int in_y2 = min(in_y1 + 1, PADDED_Y - 1);
    int in_x1 = min((int)in_x, PADDED_X - 1);
    int in_x2 = min(in_x1 + 1, PADDED_X - 1);

    const ACCUMULATOR_TYPE dx1 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x1)) : 0.5f;
    const ACCUMULATOR_TYPE dx2 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x2)) : 0.5f;
    const ACCUMULATOR_TYPE dy1 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y1)) : 0.5f;
    const ACCUMULATOR_TYPE dy2 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y2)) : 0.5f;
#if PADDING_USED == 1
    in_y1 -= PADS_BEGIN[3];
    in_y2 -= PADS_BEGIN[3];
    in_x1 -= PADS_BEGIN[4];
    in_x2 -= PADS_BEGIN[4];

    bool tlOutOfBounds = in_y1 < 0 || in_y1 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
    bool trOutOfBounds = in_y1 < 0 || in_y1 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];
    bool blOutOfBounds = in_y2 < 0 || in_y2 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
    bool brOutOfBounds = in_y2 < 0 || in_y2 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];

    unroll_for(int in_f = 0; in_f < OUTPUT_FEATURE_NUM; in_f++) {
        INPUT0_TYPE top_left = tlOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, in_f, in_y1, in_x1)];
        INPUT0_TYPE top_right = trOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, in_f, in_y1, in_x2)];
        INPUT0_TYPE bottom_left = blOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, in_f, in_y2, in_x1)];
        INPUT0_TYPE bottom_right = brOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, in_f, in_y2, in_x2)];

#else
    unroll_for(int in_f = 0; in_f < OUTPUT_FEATURE_NUM; in_f++) {
        INPUT0_TYPE top_left = input[INPUT0_GET_INDEX(batch, in_f, in_y1, in_x1)];
        INPUT0_TYPE top_right = input[INPUT0_GET_INDEX(batch, in_f, in_y1, in_x2)];
        INPUT0_TYPE bottom_left = input[INPUT0_GET_INDEX(batch, in_f, in_y2, in_x1)];
        INPUT0_TYPE bottom_right = input[INPUT0_GET_INDEX(batch, in_f, in_y2, in_x2)];
#endif

        ACCUMULATOR_TYPE interp_val = TO_ACCUMULATOR_TYPE(dx2 * dy2 * top_left) +
                                      TO_ACCUMULATOR_TYPE(dx1 * dy2 * top_right) +
                                      TO_ACCUMULATOR_TYPE(dx2 * dy1 * bottom_left) +
                                      TO_ACCUMULATOR_TYPE(dx1 * dy1 * bottom_right);

#if HAS_FUSED_OPS
        #define OF_ID (in_f)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        #undef OF_ID
#else
        OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(interp_val), ACTIVATION_PARAMS);
#endif
        output[OUTPUT_GET_INDEX(batch, in_f, oy, ox)] = res;
    }
#endif // #if OUTPUT_DIMS <= 4

#if OUTPUT_DIMS == 5
    const int ox = get_global_id(0);
    const int oy = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    const int oz = (int)get_global_id(1) / OUTPUT_SIZE_Y;

    const int feature = (int)get_global_id(2) % OUTPUT_FEATURE_NUM;
    const int batch = (int)get_global_id(2) / OUTPUT_FEATURE_NUM;

    const int PADDED_Z = in_size[2] + PADS_BEGIN[2] + PADS_END[2];
    const int PADDED_Y = in_size[3] + PADS_BEGIN[3] + PADS_END[3];
    const int PADDED_X = in_size[4] + PADS_BEGIN[4] + PADS_END[4];
    const float ix = FUNC_CALL(get_original_coordinate)(ox, SCALES[4], out_size[4], PADDED_X);
    const float iy = FUNC_CALL(get_original_coordinate)(oy, SCALES[3], out_size[3], PADDED_Y);
    const float iz = FUNC_CALL(get_original_coordinate)(oz, SCALES[2], out_size[2], PADDED_Z);

    float in_z = fmax(0, fmin(iz, PADDED_Z - 1));
    float in_y = fmax(0, fmin(iy, PADDED_Y - 1));
    float in_x = fmax(0, fmin(ix, PADDED_X - 1));

    int in_z1 = min((int)in_z, PADDED_Z - 1);
    int in_z2 = min(in_z1 + 1, PADDED_Z - 1);
    int in_y1 = min((int)in_y, PADDED_Y - 1);
    int in_y2 = min(in_y1 + 1, PADDED_Y - 1);
    int in_x1 = min((int)in_x, PADDED_X - 1);
    int in_x2 = min(in_x1 + 1, PADDED_X - 1);

    const ACCUMULATOR_TYPE dx1 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x1)) : 0.5f;
    const ACCUMULATOR_TYPE dx2 = (in_x1 != in_x2) ? TO_ACCUMULATOR_TYPE(fabs(in_x - in_x2)) : 0.5f;
    const ACCUMULATOR_TYPE dy1 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y1)) : 0.5f;
    const ACCUMULATOR_TYPE dy2 = (in_y1 != in_y2) ? TO_ACCUMULATOR_TYPE(fabs(in_y - in_y2)) : 0.5f;
    const ACCUMULATOR_TYPE dz1 = (in_z1 != in_z2) ? TO_ACCUMULATOR_TYPE(fabs(in_z - in_z1)) : 0.5f;
    const ACCUMULATOR_TYPE dz2 = (in_z1 != in_z2) ? TO_ACCUMULATOR_TYPE(fabs(in_z - in_z2)) : 0.5f;

#if PADDING_USED == 1
    in_z1 -= PADS_BEGIN[2];
    in_z2 -= PADS_BEGIN[2];
    in_y1 -= PADS_BEGIN[3];
    in_y2 -= PADS_BEGIN[3];
    in_x1 -= PADS_BEGIN[4];
    in_x2 -= PADS_BEGIN[4];

    bool BackTopLOutOfBounds = in_z1 < 0 || in_z1 >= in_size[2] || in_y1 < 0 || in_y1 >= in_size[3] || in_x1 < 0|| in_x1 >= in_size[4];
    bool BackTopROutOfBounds = in_z1 < 0 || in_z1 >= in_size[2] || in_y1 < 0 || in_y1 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];
    bool BackBottomLOutOfBounds = in_z1 < 0 || in_z1 >= in_size[2] || in_y2 < 0 || in_y2 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
    bool BackBottomROutOfBounds = in_z1 < 0 || in_z1 >= in_size[2] || in_y2 < 0 || in_y2 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];

    bool FrontTopLOutOfBounds = in_z2 < 0 || in_z2 >= in_size[2] || in_y1 < 0 || in_y1 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
    bool FrontTopROutOfBounds = in_z2 < 0 || in_z2 >= in_size[2] || in_y1 < 0 || in_y1 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];
    bool FrontBottomLOutOfBounds = in_z2 < 0 || in_z2 >= in_size[2] || in_y2 < 0 || in_y2 >= in_size[3] || in_x1 < 0 || in_x1 >= in_size[4];
    bool FrontBottomROutOfBounds = in_z2 < 0 || in_z2 >= in_size[2] || in_y2 < 0 || in_y2 >= in_size[3] || in_x2 < 0 || in_x2 >= in_size[4];

    OUTPUT_TYPE x111 = BackTopLOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y1, in_x1)];
    OUTPUT_TYPE x211 = BackTopROutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y1, in_x2)];
    OUTPUT_TYPE x121 = BackBottomLOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y2, in_x1)];
    OUTPUT_TYPE x221 = BackBottomROutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y2, in_x2)];
    OUTPUT_TYPE x112 = FrontTopLOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y1, in_x1)];
    OUTPUT_TYPE x212 = FrontTopROutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y1, in_x2)];
    OUTPUT_TYPE x122 = FrontBottomLOutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y2, in_x1)];
    OUTPUT_TYPE x222 = FrontBottomROutOfBounds ? INPUT0_VAL_ZERO : input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y2, in_x2)];
#else
    OUTPUT_TYPE x111 = input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y1, in_x1)];
    OUTPUT_TYPE x211 = input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y1, in_x2)];
    OUTPUT_TYPE x121 = input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y2, in_x1)];
    OUTPUT_TYPE x221 = input[INPUT0_GET_INDEX(batch, feature, in_z1, in_y2, in_x2)];
    OUTPUT_TYPE x112 = input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y1, in_x1)];
    OUTPUT_TYPE x212 = input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y1, in_x2)];
    OUTPUT_TYPE x122 = input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y2, in_x1)];
    OUTPUT_TYPE x222 = input[INPUT0_GET_INDEX(batch, feature, in_z2, in_y2, in_x2)];
#endif

    ACCUMULATOR_TYPE interp_val = dx2 * dy2 * dz2 * x111 + dx1 * dy2 * dz2 * x211;
    interp_val += dx2 * dy1 * dz2 * x121 + dx1 * dy1 * dz2 * x221;
    interp_val += dx2 * dy2 * dz1 * x112 + dx1 * dy2 * dz1 * x212;
    interp_val += dx2 * dy1 * dz1 * x122 + dx1 * dy1 * dz1 * x222;

#if HAS_FUSED_OPS
        #define OF_ID (feature)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        #undef OF_ID
#else
        OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(interp_val), ACTIVATION_PARAMS);
#endif
    output[OUTPUT_GET_INDEX(batch, feature, oz, oy, ox)] = res;
#endif // #if OUTPUT_DIMS == 5

#elif defined(SAMPLE_TYPE_INTERP) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int feature = 0;
    const int batch = get_global_id(2);
    const float ix = FUNC_CALL(get_original_coordinate)(ox, SCALES[4], OUTPUT_SIZE_X, in_size[4]);
    const float iy = FUNC_CALL(get_original_coordinate)(oy, SCALES[3], OUTPUT_SIZE_Y, in_size[3]);

#ifdef LEFTOVERS
    if (ox >= OUTPUT_SIZE_X)
        return;
#endif

    const int top_y_index    = (int)(floor(iy));
    const int bottom_y_index = min((int)ceil(iy), in_size[3] - 1);
    const int left_x_index   = (int)(floor(ix));
    const int right_x_index  = min((int)ceil(ix), in_size[4] - 1);

    const ACCUMULATOR_TYPE dx = TO_ACCUMULATOR_TYPE(ix - left_x_index);
    const ACCUMULATOR_TYPE dy = TO_ACCUMULATOR_TYPE(iy - top_y_index);

    unroll_for(int in_f = 0; in_f < OUTPUT_FEATURE_NUM; in_f++) {
        INPUT0_TYPE top_left = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, left_x_index)];
        INPUT0_TYPE top_right = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, right_x_index)];
        INPUT0_TYPE bottom_left = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, left_x_index)];
        INPUT0_TYPE bottom_right = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, right_x_index)];

        ACCUMULATOR_TYPE top = TO_ACCUMULATOR_TYPE(top_left) + (TO_ACCUMULATOR_TYPE(top_right) - TO_ACCUMULATOR_TYPE(top_left)) * dx;
        ACCUMULATOR_TYPE bottom = TO_ACCUMULATOR_TYPE(bottom_left) + (TO_ACCUMULATOR_TYPE(bottom_right) - TO_ACCUMULATOR_TYPE(bottom_left)) * dx;

        ACCUMULATOR_TYPE interp_val = top + (bottom - top) * dy;

#if HAS_FUSED_OPS
        #define OF_ID (in_f)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        #undef OF_ID
#else
        OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(interp_val), ACTIVATION_PARAMS);
#endif
        output[OUTPUT_GET_INDEX(batch, in_f, oy, ox)] = res;
    }
#elif defined(SAMPLE_TYPE_CAFFE_INTERP) // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    const int ox = (int)get_global_id(0) % OUTPUT_SIZE_X;
    const int oy = (int)get_global_id(0) / OUTPUT_SIZE_X;
    const int feature_block_nun = get_global_id(1);
    const int feature = feature_block_nun * FEATURE_BLOCK_SIZE;
#if OUTPUT_DIMS <= 4
    const int batch = get_global_id(2);
    const int oz = 0;
#else
    const int batch = (int)get_global_id(2) % OUTPUT_BATCH_NUM;
    const int oz    = (int)get_global_id(2) / OUTPUT_BATCH_NUM;
#endif

    ACCUMULATOR_TYPE i_b = AXES_USED[0] ? FUNC_CALL(get_original_coordinate)(batch, SCALES[0], out_size[0], PADDED_B) : batch;
    ACCUMULATOR_TYPE i_f = AXES_USED[1] ? FUNC_CALL(get_original_coordinate)(feature, SCALES[1], out_size[1], PADDED_F) : feature;
    ACCUMULATOR_TYPE i_x = AXES_USED[4] ? FUNC_CALL(get_original_coordinate)(ox, SCALES[4], out_size[4], PADDED_X) : ox;
    ACCUMULATOR_TYPE i_y = AXES_USED[3] ? FUNC_CALL(get_original_coordinate)(oy, SCALES[3], out_size[3], PADDED_Y) : oy;
    ACCUMULATOR_TYPE i_z = AXES_USED[2] ? FUNC_CALL(get_original_coordinate)(oz, SCALES[2], out_size[2], PADDED_Z) : oz;
#if PADDING_USED == 1
    i_b -= PADS_BEGIN[0];
    i_f -= PADS_BEGIN[1];
    i_z -= PADS_BEGIN[2];
    i_y -= PADS_BEGIN[3];
    i_x -= PADS_BEGIN[4];
#endif

    const int ib_r = (int)i_b;
    const int if_r = (int)i_f;
    const int ix_r = (int)i_x;
    const int iy_r = (int)i_y;
    const int iz_r = (int)i_z;

#if ANTIALIAS == 1
    const ACCUMULATOR_TYPE ab = 1.0f / SCALES[0];
    const ACCUMULATOR_TYPE af = 1.0f / SCALES[1];
    const ACCUMULATOR_TYPE ax = 1.0f / SCALES[4];
    const ACCUMULATOR_TYPE ay = 1.0f / SCALES[3];
    const ACCUMULATOR_TYPE az = 1.0f / SCALES[2];
#else
    const ACCUMULATOR_TYPE ab = 1.0f;
    const ACCUMULATOR_TYPE af = 1.0f;
    const ACCUMULATOR_TYPE ax = 1.0f;
    const ACCUMULATOR_TYPE ay = 1.0f;
    const ACCUMULATOR_TYPE az = 1.0f;
#endif
    const int rb = (SCALES[0] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ab);
    const int rf = (SCALES[1] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / af);
    const int rx = (SCALES[4] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ax);
    const int ry = (SCALES[3] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ay);
    const int rz = (SCALES[2] < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / az);

    int const b_init = max(-PADS_BEGIN[0], ib_r - rb);
    int const f_init = max(-PADS_BEGIN[1], if_r - rf);
    int const y_init = max(-PADS_BEGIN[3], iy_r - ry);
    int const x_init = max(-PADS_BEGIN[4], ix_r - rx);
    int const z_init = max(-PADS_BEGIN[2], iz_r - rz);
    int const b_max = min(PADS_END[0] + INPUT0_BATCH_NUM, ib_r + rb + 1);
    int const f_max = min(PADS_END[1] + INPUT0_FEATURE_NUM, if_r + rf + 1);
    int const y_max = min(PADS_END[3] + INPUT0_SIZE_Y, iy_r + ry + 1);
    int const x_max = min(PADS_END[4] + INPUT0_SIZE_X, ix_r + rx + 1);
    int const z_max = min(PADS_END[2] + INPUT0_SIZE_Z, iz_r + rz + 1);
#ifndef LEFTOVERS
    const int fp_max = FEATURE_BLOCK_SIZE;
#else
    const int fp_max = min(FEATURE_BLOCK_SIZE, FEATURE_LEFTOVER);
#endif
    ACCUMULATOR_TYPE sum[fp_max] = {0};
    ACCUMULATOR_TYPE wsum[fp_max] = {0};

    unroll_for(int b = b_init; b < b_max; b++) {
        unroll_for(int f = f_init; f < f_max; f++) {
            unroll_for(int z = z_init; z < z_max; z++) {
                unroll_for(int y = y_init; y < y_max; y++) {
                    unroll_for(int x = x_init; x < x_max; x++) {
                        unroll_for(int fp = 0; fp < fp_max; fp++) {
#if PADDING_USED == 1
                            bool isOutOfBounds = b < 0 || f < 0 || z < 0 || y < 0 || x < 0 ||
                                                 b >= in_size[0] || f >= in_size[1] || z >= in_size[2] ||
                                                 y >= in_size[3] || x >= in_size[4];
#endif

                            ACCUMULATOR_TYPE db = i_b - b;
                            ACCUMULATOR_TYPE df = i_f - f;
                            ACCUMULATOR_TYPE dx = i_x - x;
                            ACCUMULATOR_TYPE dy = i_y - y;
                            ACCUMULATOR_TYPE dz = i_z - z;
#if ANTIALIAS == 1
                            ACCUMULATOR_TYPE w = ab * TRIANGLE_COEFF(ab * db) *
                                                 af * TRIANGLE_COEFF(af * df) *
                                                 ax * TRIANGLE_COEFF(ax * dx) *
                                                 ay * TRIANGLE_COEFF(ay * dy) *
                                                 az * TRIANGLE_COEFF(az * dz);
#else
                            ACCUMULATOR_TYPE w = TRIANGLE_COEFF(db) *
                                                 TRIANGLE_COEFF(df) *
                                                 TRIANGLE_COEFF(dx) *
                                                 TRIANGLE_COEFF(dy) *
                                                 TRIANGLE_COEFF(dz);
#endif
                            if (w != 0 && f + fp < INPUT0_FEATURE_NUM) {
                                wsum[fp] += w;
#if PADDING_USED == 1
                                if (!isOutOfBounds)
#endif
                                    sum[fp] += w * TO_ACCUMULATOR_TYPE(input[FUNC_CALL(get_input_index)(b, f + fp, 0, z, y, x)]);
                            }
                        }
                    }
                }
            }
        }
    }
    unroll_for (int f = 0; f < fp_max; f++) {
        ACCUMULATOR_TYPE interp_val = (wsum[f] == 0) ? ACCUMULATOR_VAL_ZERO : (sum[f] / wsum[f]);
#if HAS_FUSED_OPS
        #define OF_ID (feature + f)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
        #undef OF_ID
#else
        OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(interp_val), ACTIVATION_PARAMS);
#endif
        output[FUNC_CALL(get_output_index)(batch, feature + f, 0, oz, oy, ox)] = res;
    }
#endif // defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
}

#undef TRIANGLE_COEFF
