// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if defined(__fc_f16)

#define WORK_GROUP_X 64
#define VEC_SIZE 4
__attribute__ ((reqd_work_group_size(WORK_GROUP_X, 1, 1)))
KERNEL(fc_f16)(
    const __global half  *src_vector,
    __global half        *dst_vector,
    const __global half  *matrix
#if BIAS_TERM
    , const __global half  *biases
#endif
    )
{
    local half slm[WORK_GROUP_X];
    const unsigned x = get_local_id(0);
    const unsigned y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned oidx = (y / OUTPUT_SIZE_X) * OUTPUT_Y_PITCH + y % OUTPUT_SIZE_X + OUTPUT_OFFSET;
    const unsigned batch_id = 0;
#else
    const unsigned batch_id = get_global_id(2);

    const unsigned out_z = y / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const unsigned out_yx = y % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const unsigned out_y = out_yx / (OUTPUT_SIZE_X);
    const unsigned out_x = out_yx % (OUTPUT_SIZE_X);

    const unsigned oidx = batch_id * OUTPUT_BATCH_PITCH + out_z * OUTPUT_FEATURE_PITCH + out_y * OUTPUT_Y_PITCH + out_x + OUTPUT_OFFSET;
#endif

    // TODO: we need to support multi dims. currently it doesn't
    // TODO: check cases we have padding in y/z dimensions
    unsigned w = INPUT0_BATCH_PITCH;

    #if (LAST_INPUT_SIZE_DIV_4 == 0)
    w /= VEC_SIZE;
    __global const half4 *mat_read    = (__global const half4 *) (matrix);
    const int start_offset = w * y;
    const int end_offset = start_offset + w;
    #else
    __global const half4 *mat_read    = (__global const half4 *) (matrix + w * y);
    const int start_offset = 0;
    const int end_offset = start_offset + (w + VEC_SIZE - 1) / VEC_SIZE;
    #endif

    __global const half4 *src_read    = (__global const half4 *) (src_vector + batch_id * INPUT0_BATCH_PITCH + INPUT0_OFFSET);
    int m_offset = start_offset + x;
    int v_offset = x;
    half4 sum = (half4)(0);
    #if (LAST_INPUT_SIZE_REMAINDER == 0)
    for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
        const half4 m = mat_read[m_offset];
        const half4 v = src_read[v_offset];
        sum = mad(m, v, sum);
    }
    #else

        #if (LAST_INPUT_SIZE_DIV_4 == 0)
        for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const half4 m = mat_read[m_offset];
            const half4 v = src_read[v_offset];

            sum = mad(m, v, sum);
        }
        #else
        for (; m_offset < end_offset - WORK_GROUP_X; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const half4 m = vload4(m_offset, (__global const half*)mat_read);
            const half4 v = vload4(v_offset, (__global const half*)src_read);

            sum = mad(m, v, sum);
        }

        if (m_offset < end_offset)
        {
            const half4 m = vload4(m_offset, (__global const half*)mat_read);
            const half4 v = vload4(v_offset, (__global const half*)src_read);
            if ((x + 1) == ((LAST_INPUT_SIZE_REMAINDER + VEC_SIZE - 1) / VEC_SIZE))
            {
                #if (LAST_INPUT_SIZE_DIV_4 == 3)
                    sum.xyz += m.xyz * v.xyz;
                #elif (LAST_INPUT_SIZE_DIV_4 == 2)
                    sum.xy += m.xy * v.xy;
                #else
                    sum.x += m.x * v.x;
                #endif
            }
            else
            {
                sum = mad(m, v, sum);
            }
        }
        #endif
    #endif

    slm[x] = sum.x + sum.y + sum.z + sum.w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction now
    for (int max_offset = WORK_GROUP_X / 2; max_offset > 0; max_offset >>= 1) {
        if (x < max_offset) slm[x] += slm[x + max_offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #if BIAS_TERM
    const half bias = biases[y];
    if (x == 0)
        dst_vector[oidx] = ACTIVATION(slm[0] + bias, ACTIVATION_PARAMS);
    #else
    if (x == 0)
        dst_vector[oidx] = ACTIVATION(slm[0], ACTIVATION_PARAMS);
    #endif
}
#endif


#if defined(__fc_f32)

#define WORK_GROUP_X 64
#define VEC_SIZE 4
__attribute__ ((reqd_work_group_size(WORK_GROUP_X, 1, 1)))
KERNEL(fc_f32)(
    const __global float  *src_vector,
    __global float        *dst_vector,
    const __global float  *matrix
#if BIAS_TERM
    , const __global float  *biases
#endif
    )
{
    local float slm[WORK_GROUP_X];
    const unsigned x = get_local_id(0);
    const unsigned y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned oidx = (y / OUTPUT_SIZE_X) * OUTPUT_Y_PITCH + y % OUTPUT_SIZE_X + OUTPUT_OFFSET;
    const unsigned batch_id = 0;
#else
    const unsigned batch_id = get_global_id(2);

    const unsigned out_z = y / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const unsigned out_yx = y % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const unsigned out_y = out_yx / (OUTPUT_SIZE_X);
    const unsigned out_x = out_yx % (OUTPUT_SIZE_X);

    const unsigned oidx = batch_id * OUTPUT_BATCH_PITCH + out_z * OUTPUT_FEATURE_PITCH + out_y * OUTPUT_Y_PITCH + out_x + OUTPUT_OFFSET;
#endif
    // TODO: we need to support multi dims. currently it doesn't
    // TODO: check cases we have padding in y/z dimensions
    unsigned w = INPUT0_BATCH_PITCH;

    #if BIAS_TERM
    const float bias = biases[y];
    #else
    const float bias = 0;
    #endif

    #if (LAST_INPUT_SIZE_DIV_4 == 0)
    w /= VEC_SIZE;
    __global const float4 *mat_read    = (__global const float4 *) (matrix);
    const int start_offset = w * y;
    const int end_offset = start_offset + w;
    #else
    __global const float4 *mat_read    = (__global const float4 *) (matrix + w * y);
    const int start_offset = 0;
    const int end_offset = start_offset + (w + VEC_SIZE - 1) / VEC_SIZE;
    #endif

    __global const float4 *src_read    = (__global const float4 *) (src_vector + batch_id*INPUT0_BATCH_PITCH + INPUT0_OFFSET);
    int m_offset = start_offset + x;
    int v_offset = x;
    float4 sum = (float4)(0);
    #if (LAST_INPUT_SIZE_REMAINDER == 0)
    for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
        const float4 m = mat_read[m_offset];
        const float4 v = src_read[v_offset];
        sum = mad(m, v, sum);
    }
    #else

        #if (LAST_INPUT_SIZE_DIV_4 == 0)
        for (; m_offset < end_offset; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const float4 m = mat_read[m_offset];
            const float4 v = src_read[v_offset];

            sum = mad(m, v, sum);
        }
        #else
        for (; m_offset < end_offset - WORK_GROUP_X; m_offset += WORK_GROUP_X, v_offset += WORK_GROUP_X) {
            const float4 m = mat_read[m_offset];
            const float4 v = src_read[v_offset];

            sum = mad(m, v, sum);
        }

        if (m_offset < end_offset)
        {
            const float4 m = mat_read[m_offset];
            const float4 v = src_read[v_offset];
            if ((x + 1) == ((LAST_INPUT_SIZE_REMAINDER + VEC_SIZE - 1) / VEC_SIZE))
            {
                #if (LAST_INPUT_SIZE_DIV_4 == 3)
                    sum.xyz += m.xyz * v.xyz;
                #elif (LAST_INPUT_SIZE_DIV_4 == 2)
                    sum.xy += m.xy * v.xy;
                #else
                    sum.x += m.x * v.x;
                #endif
            }
            else
            {
                sum = mad(m, v, sum);
            }
        }
        #endif
    #endif

    slm[x] = sum.x + sum.y + sum.z + sum.w;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction now
    for (int max_offset = WORK_GROUP_X / 2; max_offset > 0; max_offset >>= 1) {
        if (x < max_offset) slm[x] += slm[x + max_offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x == 0)
        dst_vector[oidx] = ACTIVATION(slm[0] + bias, ACTIVATION_PARAMS);
}
#endif
