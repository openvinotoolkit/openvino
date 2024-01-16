__kernel void custom_kernel(__global const INPUT0_TYPE* input, __global OUTPUT0_TYPE* output) {
    uint id = get_global_id(0);

    output[id] = input[id] * alpha + beta;
}
