// Variant A: convert all 16 bytes (uints 0..3) at once via char16->half16, then repack.
// RESULT: 93 mov (asym) — SAME as baseline. IGC already vectorizes the baseline the same
// way; a wider convert hint is a no-op. See sdpa-ocl-int8-perf lever-1 follow-up.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#define VAL_ZERO_POINTS 1
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(const global void* V, int ww,int hh,int pp, global half* Vz,
                int k0, global int* out) {
    int lane=get_sub_group_local_id();
    const half vz_c = Vz[lane];
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
        (global void*)V, ww,hh,pp, (int2)(0, k0), (private uint*)&vt[0]);
    char16 raw = as_char16((uint4)(vt[0],vt[1],vt[2],vt[3]));
    half16 hv = convert_half16(raw);
    half16 zp;
    #pragma unroll
    for (int i=0;i<16;i++) zp[i]=sub_group_broadcast(vz_c,i);
    half16 deq = hv - zp;
    int8 vb;
    #pragma unroll
    for (int i=0;i<8;i++) vb[i]=as_int((half2)(deq[2*i],deq[2*i+1]));
    ((global int8*)out)[lane]=vb;
}
