// Microbench: current V int8 dequant/repack for ONE cp-block (with zp).
// 8b VNNI-transform read -> char4->half4 unpack -> per-key zp broadcast -> .lo/.hi
// repack into the f16 VNNI operand vb. This is the ~93 mov/cp cost the lever-1
// investigation isolated. Baseline for the v_dequant_* variants. See sdpa-ocl-int8-perf.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#define VAL_ZERO_POINTS 1
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(const global void* V, int w,int h,int p, global half* Vz, int k0, global int* out) {
    int lane=get_sub_group_local_id();
    const int cp=0, cd=0, sg_j0_sv=0;
    const half vz_c = Vz[lane];
    int8 vb;
    uint vt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c(
        (global void*)V, w,h,p, (int2)(sg_j0_sv+cd*16, k0+cp*16), (private uint*)&vt[0]);
    #pragma unroll
    for (int u=0; u<4; ++u) {
        const half4 raw4 = convert_half4(as_char4(vt[u]));
        const int k0r=u*4;
        const half4 zp4=(half4)(sub_group_broadcast(vz_c,k0r+0),sub_group_broadcast(vz_c,k0r+1),
                                sub_group_broadcast(vz_c,k0r+2),sub_group_broadcast(vz_c,k0r+3));
        const half4 deq4=raw4-zp4;
        vb[u*2+0]=as_int(deq4.lo); vb[u*2+1]=as_int(deq4.hi);
    }
    ((global int8*)out)[lane]=vb;
}
