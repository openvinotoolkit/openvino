// Copies port 1 only. Port 0 is declared BFYX and port 1 YXFB while both read the same
// producer, so the two ports must get their own pre-reorders; if port 1 reused port 0's
// BFYX reorder the copy would come out permuted.
__kernel void custom_kernel_second_port_copy(__global const INPUT0_TYPE* input0,
                                             __global const INPUT1_TYPE* input1,
                                             __global OUTPUT0_TYPE* output) {
    const int i = get_global_id(0);
    output[i] = input1[i];
}
