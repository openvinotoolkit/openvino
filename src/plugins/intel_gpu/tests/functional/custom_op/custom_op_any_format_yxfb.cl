// Flat elementwise copy. The XML declares input port 0 as YXFB, so the
// plugin reorders the input to YXFB and this kernel therefore writes YXFB-ordered data.
// The output is declared ANY, which historically means "inherit the first input's
// format" -- i.e. YXFB -- so the plugin must reorder YXFB->BFYX for the Result.
__kernel void custom_kernel_yxfb_copy(__global const INPUT0_TYPE* input,
                                      __global OUTPUT0_TYPE* output) {
    const int i = get_global_id(0);
    output[i] = input[i] + (OUTPUT0_TYPE)1;
}
