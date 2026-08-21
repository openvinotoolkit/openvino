// Flat copy for the dynamic-shape ANY-output case.
__kernel void custom_kernel_dyn_copy(__global const INPUT0_TYPE* input,
                                     __global OUTPUT0_TYPE* output) {
    const int i = get_global_id(0);
    output[i] = input[i] + (OUTPUT0_TYPE)1;
}
