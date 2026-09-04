__kernel void custom_kernel_two_inputs(__global const INPUT0_TYPE* a,
                                       __global const INPUT1_TYPE* b,
                                       __global OUTPUT0_TYPE* output) {
    const uint id = get_global_id(0);

    output[id] = a[id] + b[id];
}
