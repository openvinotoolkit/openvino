// Variant E: char16->half16->as_int8 reinterpret (no manual .lo/.hi per-slot repack),
// relying on keys being in-order so half16 IS the VNNI-2 operand. Symmetric.
// RESULT: 68 mov — same as B/C. The int8->f16 WIDENING (256B->512B reg footprint) is
// unavoidable physical movement; logical order was already correct. See sdpa-ocl-int8-perf.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(const global void* V, int ww,int hh,int pp, int k0, global int* out) {
    int lane=get_sub_group_local_id();
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void*)V, ww,hh,pp,(int2)(0,k0),(private uint*)&vt[0]);
    char16 raw = as_char16((uint4)(vt[0],vt[1],vt[2],vt[3]));
    half16 hv = convert_half16(raw);       // keys 0..15 as half, in order
    int8 vb = as_int8(hv);                  // int[p] = (hv[2p], hv[2p+1]) = keys 2p,2p+1 = VNNI-2
    ((global int8*)out)[lane]=vb;
}
