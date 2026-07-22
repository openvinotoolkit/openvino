__kernel void custom_kernel_intr_buffer(
    __global const float* input,
    __global float* output,
    __global float* internal_buf
) {
    uint id = get_global_id(0);

    // store intermediate
    internal_buf[id] = input[id];

    // use it later
    output[id] = internal_buf[id];
}
