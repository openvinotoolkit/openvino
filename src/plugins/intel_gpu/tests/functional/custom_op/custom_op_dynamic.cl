__kernel void custom_add_kernel(
    __global const INPUT0_TYPE* inp0,
    __global OUTPUT0_TYPE* outp) {
    const uint b = (uint)get_global_id(0);
    const uint f = (uint)get_global_id(1);
    const uint y = (uint)get_global_id(2);
    #if INPUT0_DIMS_SIZE == 4
        const uint x = 0;
    #endif

    const unsigned src_index = b*INPUT0_DIMS[1]*INPUT0_DIMS[2]*INPUT0_DIMS[3] + f*INPUT0_DIMS[2]*INPUT0_DIMS[3] + y*INPUT0_DIMS[3] + x;
    const unsigned dst_index = src_index;

    outp[dst_index] = inp0[src_index] * alpha + beta;
}
