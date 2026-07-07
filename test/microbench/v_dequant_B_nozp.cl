// Variant B: NO zp (symmetric int8). Isolates the cost of the 16 zp broadcasts + subtract.
// RESULT: 68 mov vs baseline 93 (asym) => zp handling = ~25 mov/cp. This is the only real
// (but narrow: symmetric-only) win found in lever-1. See sdpa-ocl-int8-perf.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(const global void* V, int ww,int hh,int pp, int k0, global int* out) {
    int lane=get_sub_group_local_id();
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
        (global void*)V, ww,hh,pp, (int2)(0, k0), (private uint*)&vt[0]);
    int8 vb;
    #pragma unroll
    for (int u=0; u<4; ++u) {
        const half4 deq4 = convert_half4(as_char4(vt[u]));
        vb[u*2+0]=as_int(deq4.lo); vb[u*2+1]=as_int(deq4.hi);
    }
    ((global int8*)out)[lane]=vb;
}
