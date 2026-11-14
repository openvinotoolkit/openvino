#include "include/batch_headers/common.cl"

KERNEL(normalize_gpu_across_x)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    const __global SCALE_TABLE_TYPE* scale_input)
{
    const uint lane = get_local_id(0);
    const uint y = get_group_id(1);
    const uint f = get_group_id(2) % INPUT0_FEATURE_NUM;
    const uint b = get_group_id(2) / INPUT0_FEATURE_NUM;

    float sum = 0.0f;
    for (uint x = lane; x < INPUT0_SIZE_X; x += LOCAL_SIZE) {
        const float value = (float)input[INPUT0_GET_INDEX(b, f, y, x)];
        sum = fma(value, value, sum);
    }

    __local float partial_sums[LOCAL_SIZE];
    partial_sums[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = LOCAL_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial_sums[lane] += partial_sums[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float denominator = sqrt(partial_sums[0] + EPSILON);
#if SCALE_TABLE_FEATURE_NUM == 1
    const uint scale_index = 0;
#else
    const uint scale_index = SCALE_TABLE_GET_INDEX(0, f, 0, 0);
#endif
    const float scale = (float)scale_input[scale_index];
    for (uint x = lane; x < INPUT0_SIZE_X; x += LOCAL_SIZE) {
        const float result = (float)input[INPUT0_GET_INDEX(b, f, y, x)] / denominator * scale;
        const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#endif
    }
}