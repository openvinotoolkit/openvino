__kernel void custom_add_kernel(
    __global const INPUT0_TYPE* inp0,
    __global OUTPUT0_TYPE* outp) {
    int b = get_global_id(0);
    int f = get_global_id(1);
    int y = get_global_id(2);
    // shape: [-1, 1, 2]
    int id = b * 1 * 2 + f * 2 + y;
    outp[id] = inp0[id] * alpha + beta;
}
