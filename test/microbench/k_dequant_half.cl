// Microbench: K int8 dequant with scale/zp in HALF (the SLOWER alternative).
// Same as k_dequant_float.cl but scale/zp and the dequant arithmetic are half.
// Counterintuitively SLOWER: GEN forces the int-word->half conversion mov to a
// dword-strided <2> destination, so the half intermediates need extra :uw<2> repack
// movs to assemble the contiguous short8 DPAS A operand; the float path stays dense
// <1> and its muls get {Compacted} encoding. Measured: half=144 instr/93 mov vs
// float=134/83. See sdpa-ocl-int8-perf memory. Do NOT unify K to half.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#define SUBGROUP_SIZE 16
#define KEY_ZERO_POINTS 1
#define kq_sg_tile_keys 16
#define kq_key_blocks 2
#define DPAS_K 16
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void t(const global void* K,int KD_w,int KD_h,int KD_p,int db,int key_base,
                global half* Ks, global half* Kz, global int* out){
    int lane=get_sub_group_local_id();
    half k_scale_lane[kq_sg_tile_keys/SUBGROUP_SIZE];
    half k_zp_lane[kq_sg_tile_keys/SUBGROUP_SIZE];
    #pragma unroll
    for(int ii=0;ii<kq_sg_tile_keys/SUBGROUP_SIZE;++ii){ k_scale_lane[ii]=Ks[lane]; k_zp_lane[ii]=Kz[lane]; }
    ushort8 k_raw[kq_key_blocks];
    uint kt[8];
    intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void*)K,KD_w,KD_h,KD_p,(int2)(db*DPAS_K,key_base),(private uint*)&kt[0]);
    #pragma unroll
    for(int mb=0;mb<kq_key_blocks;++mb) k_raw[mb]=(ushort8)0;
    #pragma unroll
    for(int u=0;u<kq_sg_tile_keys/4;++u){
        const char4 raw4=as_char4(kt[u]);
        #pragma unroll
        for(int bb=0;bb<4;++bb){
            const int krel=u*4+bb;
            const half k_sc=sub_group_broadcast(k_scale_lane[krel/SUBGROUP_SIZE],krel%SUBGROUP_SIZE);
            const half k_zp=sub_group_broadcast(k_zp_lane[krel/SUBGROUP_SIZE],krel%SUBGROUP_SIZE);
            const half deq_k=(convert_half(raw4[bb])-k_zp)*k_sc;
            k_raw[krel/8][krel%8]=as_ushort(deq_k);
        }
    }
    ((global int8*)out)[lane]=as_int8((ushort16)(k_raw[0],k_raw[1]));
}
