// Rank-3 companion to custom_op_any_format.cl.
__kernel void custom_kernel_axis_probe_3d(__global const INPUT0_TYPE* input,
                                          __global OUTPUT0_TYPE* output) {
    const int y = get_global_id(0);
    const int f = get_global_id(1);
    const int b = get_global_id(2);

    const int fnum = OUTPUT0_DIMS[1];
    const int ynum = OUTPUT0_DIMS[2];

    output[(b * fnum + f) * ynum + y] = (OUTPUT0_TYPE)(b * 100 + f * 10 + y);
}
