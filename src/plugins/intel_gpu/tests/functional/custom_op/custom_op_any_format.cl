// Derives its output from its dispatch coordinates and the declared output width; a flat
// copy would map element i to element i whatever the layout.
__kernel void custom_kernel_axis_probe(__global const INPUT0_TYPE* input,
                                       __global OUTPUT0_TYPE* output) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int w = OUTPUT0_DIMS[3];

    output[y * w + x] = (OUTPUT0_TYPE)(y * 1000 + x);
}
