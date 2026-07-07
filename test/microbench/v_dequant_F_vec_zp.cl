// Variant F: build a half16 zp vector once, single vector subtract from half16 V. Asymmetric.
// RESULT: 93 mov (= baseline). IGC scalarizes the half16 vector subtract back into 16 adds
// because each zp element is a distinct-lane sub_group_broadcast (not a contiguous vector).
// So expressing zp as a vector doesn't reduce the broadcast cost. See sdpa-ocl-int8-perf.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#define VAL_ZERO_POINTS 1
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(const global void* V, int ww,int hh,int pp, global half* Vz, int k0, global int* out) {
    int lane=get_sub_group_local_id();
    const half vz_c = Vz[lane];  // lane=key: this lane's zp
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void*)V, ww,hh,pp,(int2)(0,k0),(private uint*)&vt[0]);
    half16 zp16;
    #pragma unroll
    for (int i=0;i<16;i++) zp16[i]=sub_group_broadcast(vz_c,i);
    char16 raw = as_char16((uint4)(vt[0],vt[1],vt[2],vt[3]));
    half16 deq = convert_half16(raw) - zp16;   // single vector subtract (IGC scalarizes it)
    int8 vb = as_int8(deq);
    ((global int8*)out)[lane]=vb;
}
