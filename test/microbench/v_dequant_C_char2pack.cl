// Variant C: char2->half2 direct pack (no half4 .lo/.hi intermediate). Symmetric.
// RESULT: 68 mov — same as B; the repack itself is IGC's floor, changing the pack
// expression doesn't help. See sdpa-ocl-int8-perf lever-1.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(const global void* V, int ww,int hh,int pp, int k0, global int* out) {
    int lane=get_sub_group_local_id();
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
        (global void*)V, ww,hh,pp, (int2)(0, k0), (private uint*)&vt[0]);
    char4 c0=as_char4(vt[0]), c1=as_char4(vt[1]), c2=as_char4(vt[2]), c3=as_char4(vt[3]);
    int8 vb;
    vb[0]=as_int(convert_half2(c0.lo)); vb[1]=as_int(convert_half2(c0.hi));
    vb[2]=as_int(convert_half2(c1.lo)); vb[3]=as_int(convert_half2(c1.hi));
    vb[4]=as_int(convert_half2(c2.lo)); vb[5]=as_int(convert_half2(c2.hi));
    vb[6]=as_int(convert_half2(c3.lo)); vb[7]=as_int(convert_half2(c3.hi));
    ((global int8*)out)[lane]=vb;
}
