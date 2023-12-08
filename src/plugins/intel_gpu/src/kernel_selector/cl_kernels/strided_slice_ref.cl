// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#ifdef STRIDE_TYPE
inline void FUNC(get_slice_step)(OPTIONAL_SHAPE_INFO_ARG
                                 const __global STRIDE_TYPE* stride,
                                 int* step_batch, int* step_feature,
                                 int* step_w, int* step_z, int* step_y, int* step_x)
{
    const uint batch_index = 0;
    const uint feature_index = 1;
#ifdef OUTPUT_LAYOUT_BFYX
    const uint y_index = 2;
    const uint x_index = 3;
#elif OUTPUT_LAYOUT_BFZYX
    const uint z_index = 2;
    const uint y_index = 3;
    const uint x_index = 4;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w_index = 2;
    const uint z_index = 3;
    const uint y_index = 4;
    const uint x_index = 5;
#endif

    *step_batch = batch_index < STRIDE_DIMS ? stride[batch_index] : 1;
    *step_feature = feature_index < STRIDE_DIMS ? stride[feature_index] : 1;
#ifdef OUTPUT_LAYOUT_BFYX
    *step_w = 0;
    *step_z = 0;
#elif OUTPUT_LAYOUT_BFZYX
    *step_w = 0;
    *step_z = z_index < STRIDE_DIMS ? stride[z_index] : 1;
#elif OUTPUT_LAYOUT_BFWZYX
    *step_w = w_index < STRIDE_DIMS ? stride[w_index] : 1;
    *step_z = z_index < STRIDE_DIMS ? stride[z_index] : 1;
#endif
    *step_y = y_index < STRIDE_DIMS ? stride[y_index] : 1;
    *step_x = x_index < STRIDE_DIMS ? stride[x_index] : 1;
}
#endif // STRIDE_TYPE

#ifdef END_TYPE
inline int FUNC(check_end_bound)(const end_num,
                                 const uint out_num)
{
    int num;
    if (end_num < 0) {
        num = max(TO_END_TYPE(out_num) + end_num, TO_END_TYPE(0));
    } else {
        num = end_num;
    }
    num = min(num, (int)out_num);
    return num;
}

inline void FUNC(get_slice_end)(OPTIONAL_SHAPE_INFO_ARG
                                const __global END_TYPE* end,
                                int* end_batch, int* end_feature,
                                int* end_w, int* end_z, int* end_y, int* end_x)
{
    const uint out_batch_num = INPUT0_BATCH_NUM;
    const uint out_feature_num = INPUT0_FEATURE_NUM;
    const uint out_w_num = INPUT0_SIZE_W;
    const uint out_z_num = INPUT0_SIZE_Z;
    const uint out_y_num = INPUT0_SIZE_Y;
    const uint out_x_num = INPUT0_SIZE_X;
    const uint batch_index = 0;
    const uint feature_index = 1;
#ifdef OUTPUT_LAYOUT_BFYX
    const uint y_index = 2;
    const uint x_index = 3;
#elif OUTPUT_LAYOUT_BFZYX
    const uint z_index = 2;
    const uint y_index = 3;
    const uint x_index = 4;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w_index = 2;
    const uint z_index = 3;
    const uint y_index = 4;
    const uint x_index = 5;
#endif
    END_TYPE batch = batch_index < END_DIMS ? end[batch_index] : 0;
    END_TYPE feature = feature_index < END_DIMS ? end[feature_index] : 0;
#ifdef OUTPUT_LAYOUT_BFWZYX
    END_TYPE w = w_index < END_DIMS ? end[w_index] : 0;
    END_TYPE z = z_index < END_DIMS ? end[z_index] : 0;
#elif OUTPUT_LAYOUT_BFZYX
    END_TYPE z = z_index < END_DIMS ? end[z_index] : 0;
#endif
    END_TYPE y = y_index < END_DIMS ? end[y_index] : 0;
    END_TYPE x = x_index < END_DIMS ? end[x_index] : 0;

    batch = (END_BATCH == 0) ? batch : TO_END_TYPE(out_batch_num);
    feature = (END_FEATURE == 0) ? feature : TO_END_TYPE(out_feature_num);
#ifdef OUTPUT_LAYOUT_BFWZYX
    w = (END_W == 0) ? w: TO_END_TYPE(out_w_num);
    z = (END_Z == 0) ? z: TO_END_TYPE(out_z_num);
#elif OUTPUT_LAYOUT_BFZYX
    z = (END_Z == 0) ? z: TO_END_TYPE(out_z_num);
#endif
    y = (END_Y == 0) ? y : TO_END_TYPE(out_y_num);
    x = (END_X == 0) ? x : TO_END_TYPE(out_x_num);

    *end_batch = FUNC_CALL(check_end_bound)(batch, out_batch_num);
    *end_feature = FUNC_CALL(check_end_bound)(feature, out_feature_num);
#ifdef OUTPUT_LAYOUT_BFYX
    *end_w = 0;
    *end_z = 0;
#elif OUTPUT_LAYOUT_BFZYX
    *end_z = FUNC_CALL(check_end_bound)(z, out_z_num);
#elif OUTPUT_LAYOUT_BFWZYX
    *end_w = FUNC_CALL(check_end_bound)(w, out_w_num);
    *end_z = FUNC_CALL(check_end_bound)(z, out_z_num);
#endif
    *end_y = FUNC_CALL(check_end_bound)(y, out_y_num);
    *end_x = FUNC_CALL(check_end_bound)(x, out_x_num);
}

inline void FUNC(check_negative_stride)(OPTIONAL_SHAPE_INFO_ARG
                                        const __global END_TYPE* end,
                                        const int steps_batch, const int steps_feature,
                                        const int steps_w, const int steps_z, const int steps_y, const int steps_x,
                                        int* begin_batch, int* begin_feature,
                                        int* begin_w, int* begin_z, int* begin_y, int* begin_x)
{
    bool is_negative = (steps_batch < 0) || (steps_feature < 0) || (steps_w < 0) || (steps_z < 0) || (steps_y < 0) || (steps_x < 0);
    if (is_negative) {
        int end_batch, end_feature, end_w, end_z, end_y, end_x;
        FUNC_CALL(get_slice_end)(OPTIONAL_SHAPE_INFO_TENSOR end, &end_batch, &end_feature, &end_w, &end_z, &end_y, &end_x);
        const int slice_end_batch = end_batch;
        const int slice_end_feature = end_feature;
        const int slice_end_w = end_w;
        const int slice_end_z = end_z;
        const int slice_end_y = end_y;
        const int slice_end_x = end_x;

        if ((steps_batch < 0) && (*begin_batch <= slice_end_batch))
            *begin_batch = slice_end_batch - 1;
        if ((steps_feature < 0) && (*begin_feature <= slice_end_feature))
            *begin_feature = slice_end_feature - 1;
        if ((steps_w < 0) && (*begin_w <= slice_end_w))
            *begin_w = slice_end_w - 1;
        if ((steps_z < 0) && (*begin_z <= slice_end_z))
            *begin_z = slice_end_z - 1;
        if ((steps_y < 0) && (*begin_y <= slice_end_y))
            *begin_y = slice_end_y - 1;
        if ((steps_x < 0) && (*begin_x <= slice_end_x))
            *begin_x = slice_end_x - 1;
    }
}
#else // END_TYPE
inline void FUNC(check_negative_stride)(const int steps_batch, const int steps_feature,
                                        const int steps_w, const int steps_z, const int steps_y, const int steps_x,
                                        int* begin_batch, int* begin_feature,
                                        int* begin_w, int* begin_z, int* begin_y, int* begin_x)
{
    const int slice_end_batch = SLICE_END_BATCH;
    const int slice_end_feature = SLICE_END_FEATURE;
    const int slice_end_w = SLICE_END_W;
    const int slice_end_z = SLICE_END_Z;
    const int slice_end_y = SLICE_END_Y;
    const int slice_end_x = SLICE_END_X;

    if ((steps_batch < 0) && (*begin_batch <= slice_end_batch))
        *begin_batch = slice_end_batch - 1;
    if ((steps_feature < 0) && (*begin_feature <= slice_end_feature))
        *begin_feature = slice_end_feature - 1;
    if ((steps_w < 0) && (*begin_w <= slice_end_w))
        *begin_w = slice_end_w - 1;
    if ((steps_z < 0) && (*begin_z <= slice_end_z))
        *begin_z = slice_end_z - 1;
    if ((steps_y < 0) && (*begin_y <= slice_end_y))
        *begin_y = slice_end_y - 1;
    if ((steps_x < 0) && (*begin_x <= slice_end_x))
        *begin_x = slice_end_x - 1;
}
#endif // END_TYPE

#ifdef BEGIN_TYPE
inline int FUNC(check_begin_bound)(BEGIN_TYPE begin_num,
                                   const uint out_num)
{
    int num;
    if (begin_num < 0) {
        num = max(TO_BEGIN_TYPE(out_num) + begin_num, TO_BEGIN_TYPE(0));
    } else {
        num = begin_num;
    }
    num = min(num, (int)out_num);
    return num;
}

inline void FUNC(get_slice_begin)(OPTIONAL_SHAPE_INFO_ARG
                                  const __global BEGIN_TYPE* begin,
                                  int* begin_batch, int* begin_feature,
                                  int* begin_w, int* begin_z, int* begin_y, int* begin_x)
{
    const uint out_batch_num = INPUT0_BATCH_NUM;
    const uint out_feature_num = INPUT0_FEATURE_NUM;
    const uint out_w_num = INPUT0_SIZE_W;
    const uint out_z_num = INPUT0_SIZE_Z;
    const uint out_y_num = INPUT0_SIZE_Y;
    const uint out_x_num = INPUT0_SIZE_X;
    const uint batch_index = 0;
    const uint feature_index = 1;
#ifdef OUTPUT_LAYOUT_BFYX
    const uint y_index = 2;
    const uint x_index = 3;
#elif OUTPUT_LAYOUT_BFZYX
    const uint z_index = 2;
    const uint y_index = 3;
    const uint x_index = 4;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w_index = 2;
    const uint z_index = 3;
    const uint y_index = 4;
    const uint x_index = 5;
#endif
    BEGIN_TYPE batch = batch_index < BEGIN_DIMS ? begin[batch_index] : 0;
    BEGIN_TYPE feature = feature_index < BEGIN_DIMS ? begin[feature_index] : 0;
#ifdef OUTPUT_LAYOUT_BFWZYX
    BEGIN_TYPE w = w_index < BEGIN_DIMS ? begin[w_index] : 0;
    BEGIN_TYPE z = z_index < BEGIN_DIMS ? begin[z_index] : 0;
#elif OUTPUT_LAYOUT_BFZYX
    BEGIN_TYPE z = z_index < BEGIN_DIMS ? begin[z_index] : 0;
#endif
    BEGIN_TYPE y = y_index < BEGIN_DIMS ? begin[y_index] : 0;
    BEGIN_TYPE x = x_index < BEGIN_DIMS ? begin[x_index] : 0;

    batch = (BEGIN_BATCH == 0) ? batch : 0;
    feature = (BEGIN_FEATURE == 0) ? feature : 0;
#ifdef OUTPUT_LAYOUT_BFWZYX
    w = (BEGIN_W == 0) ? w: 0;
    z = (BEGIN_Z == 0) ? z: 0;
#elif OUTPUT_LAYOUT_BFZYX
    z = (BEGIN_Z == 0) ? z: 0;
#endif
    y = (BEGIN_Y == 0) ? y : 0;
    x = (BEGIN_X == 0) ? x : 0;

    *begin_batch = FUNC_CALL(check_begin_bound)(batch, out_batch_num);
    *begin_feature = FUNC_CALL(check_begin_bound)(feature, out_feature_num);
#ifdef OUTPUT_LAYOUT_BFYX
    *begin_w = 0;
    *begin_z = 0;
#elif OUTPUT_LAYOUT_BFZYX
    *begin_w = 0;
    *begin_z = FUNC_CALL(check_begin_bound)(z, out_z_num);
#elif OUTPUT_LAYOUT_BFWZYX
    *begin_w = FUNC_CALL(check_begin_bound)(w, out_w_num);
    *begin_z = FUNC_CALL(check_begin_bound)(z, out_z_num);
#endif
    *begin_y = FUNC_CALL(check_begin_bound)(y, out_y_num);
    *begin_x = FUNC_CALL(check_begin_bound)(x, out_x_num);
}
#endif // BEGIN_TYPE

KERNEL(strided_slice_ref)(OPTIONAL_SHAPE_INFO_ARG
                          const __global INPUT0_TYPE* input,
#ifdef BEGIN_TYPE
                          const __global BEGIN_TYPE* begin,
#endif
#ifdef END_TYPE
                          const __global END_TYPE* end,
#endif
#ifdef STRIDE_TYPE
                          const __global STRIDE_TYPE* stride,
#endif
                          __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
#ifdef STRIDE_TYPE
    int step_batch, step_feature, step_w, step_z, step_y, step_x;
    FUNC_CALL(get_slice_step)(OPTIONAL_SHAPE_INFO_TENSOR stride, &step_batch, &step_feature, &step_w, &step_z, &step_y, &step_x);
    const int slice_steps_batch = step_batch;
    const int slice_steps_feature = step_feature;
    const int slice_steps_w = step_w;
    const int slice_steps_z = step_z;
    const int slice_steps_y = step_y;
    const int slice_steps_x = step_x;
#else // STRIDE_TYPE
    const int slice_steps_batch = SLICE_STEPS_BATCH;
    const int slice_steps_feature = SLICE_STEPS_FEATURE;
    const int slice_steps_w = SLICE_STEPS_W;
    const int slice_steps_z = SLICE_STEPS_Z;
    const int slice_steps_y = SLICE_STEPS_Y;
    const int slice_steps_x = SLICE_STEPS_X;
#endif // STRIDE_TYPE
#ifdef BEGIN_TYPE
    int begin_batch, begin_feature, begin_w, begin_z, begin_y, begin_x;
    FUNC_CALL(get_slice_begin)(OPTIONAL_SHAPE_INFO_TENSOR begin, &begin_batch, &begin_feature, &begin_w, &begin_z, &begin_y, &begin_x);
#ifdef END_TYPE
    FUNC_CALL(check_negative_stride)(OPTIONAL_SHAPE_INFO_TENSOR end, slice_steps_batch, slice_steps_feature, slice_steps_w, slice_steps_z, slice_steps_y, slice_steps_x, &begin_batch, &begin_feature, &begin_w, &begin_z, &begin_y, &begin_x);
#else // END_TYPE
    FUNC_CALL(check_negative_stride)(slice_steps_batch, slice_steps_feature, slice_steps_w, slice_steps_z, slice_steps_y, slice_steps_x, &begin_batch, &begin_feature, &begin_w, &begin_z, &begin_y, &begin_x);
#endif // END_TYPE
    const int slice_begin_batch = begin_batch;
    const int slice_begin_feature = begin_feature;
    const int slice_begin_w = begin_w;
    const int slice_begin_z = begin_z;
    const int slice_begin_y = begin_y;
    const int slice_begin_x = begin_x;
#else // BEGIN_TYPE
    const int slice_begin_batch = SLICE_BEGIN_BATCH;
    const int slice_begin_feature = SLICE_BEGIN_FEATURE;
    const int slice_begin_w = SLICE_BEGIN_W;
    const int slice_begin_z = SLICE_BEGIN_Z;
    const int slice_begin_y = SLICE_BEGIN_Y;
    const int slice_begin_x = SLICE_BEGIN_X;
#endif // BEGIN_TYPE

#if NEW_AXIS_MODE
    // If NEW_AXIS_MODE that just copy input to output
#ifdef OUTPUT_LAYOUT_BFYX
    const uint w_input = 0;
    const uint z_input = 0;
    const uint y_input = (uint)get_global_id(2) / INPUT0_SIZE_X;
    const uint x_input = (uint)get_global_id(2) % INPUT0_SIZE_X;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w_input = 0;
    const uint yx_input = (uint)get_global_id(2) % (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint z_input = (uint)get_global_id(2) / (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint y_input = yx_input / INPUT0_SIZE_X;
    const uint x_input = yx_input % INPUT0_SIZE_X;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint zyx_input = (uint)get_global_id(2) % (INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z);
    const uint w_input = (uint)get_global_id(2) / (INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z);
    const uint z_input = zyx_input / (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint yx_input = zyx_input % (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint y_input = yx_input / INPUT0_SIZE_X;
    const uint x_input = yx_input % INPUT0_SIZE_X;
#endif
    const uint input_index = INPUT0_OFFSET +
        batch * INPUT0_BATCH_PITCH +
        feature * INPUT0_FEATURE_PITCH +
        w_input * INPUT0_W_PITCH +
        z_input * INPUT0_Z_PITCH +
        y_input * INPUT0_Y_PITCH +
        x_input * INPUT0_X_PITCH;
    output[input_index] = input[input_index];
#else // NEW_AXIS_MODE
#ifdef OUTPUT_LAYOUT_BFYX
    const uint w = 0;
    const uint z = 0;
    const uint y = get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = get_global_id(2) % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w = 0;
    const uint yx = get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint zyx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint w = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint z = zyx / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = zyx % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#endif

#if SHRINK_MODE
    const uint in_indices[] = {INPUT_INDICES_ORDER};
    const uint input_index = INPUT0_OFFSET +
        (slice_begin_batch + in_indices[0] * slice_steps_batch) * INPUT0_BATCH_PITCH +
        (slice_begin_feature + in_indices[1] * slice_steps_feature) * INPUT0_FEATURE_PITCH +
    #if INPUT0_LAYOUT_BFWZYX
        (slice_begin_w + in_indices[2] * slice_steps_w) * INPUT0_W_PITCH +
        (slice_begin_z + in_indices[3] * slice_steps_z) * INPUT0_Z_PITCH +
        (slice_begin_y + in_indices[4] * slice_steps_y) * INPUT0_Y_PITCH +
        (slice_begin_x + in_indices[5] * slice_steps_x) * INPUT0_X_PITCH;
    #elif INPUT0_LAYOUT_BFZYX
        (slice_begin_z + in_indices[2] * slice_steps_z) * INPUT0_Z_PITCH +
        (slice_begin_y + in_indices[3] * slice_steps_y) * INPUT0_Y_PITCH +
        (slice_begin_x + in_indices[4] * slice_steps_x) * INPUT0_X_PITCH;
    #else
        (slice_begin_y + in_indices[2] * slice_steps_y) * INPUT0_Y_PITCH +
        (slice_begin_x + in_indices[3] * slice_steps_x) * INPUT0_X_PITCH;
    #endif
#else // SHRINK_MODE
    const uint input_index = INPUT0_OFFSET +
            (slice_begin_batch + batch * slice_steps_batch) * INPUT0_BATCH_PITCH +
            (slice_begin_feature + feature * slice_steps_feature) * INPUT0_FEATURE_PITCH +
            (slice_begin_w + w * slice_steps_w) * INPUT0_W_PITCH +
            (slice_begin_z + z * slice_steps_z) * INPUT0_Z_PITCH +
            (slice_begin_y + y * slice_steps_y) * INPUT0_Y_PITCH +
            (slice_begin_x + x * slice_steps_x) * INPUT0_X_PITCH;
#endif // SHRINK_MODE

    const uint output_index = OUTPUT_OFFSET +
            batch * OUTPUT_BATCH_PITCH +
            feature * OUTPUT_FEATURE_PITCH +
            w * OUTPUT_W_PITCH +
            z * OUTPUT_Z_PITCH +
            y * OUTPUT_Y_PITCH +
            x * OUTPUT_X_PITCH;

#if HAS_FUSED_OPS
    INPUT0_TYPE input_data = input[input_index];
    FUSED_OPS;
    output[output_index] = FUSED_OPS_RESULT;
#else
    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
#endif
#endif // NEW_AXIS_MODE
}
