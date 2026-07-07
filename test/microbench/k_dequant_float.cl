// Microbench: K int8 dequant with scale/zp in FLOAT (the current sdpa_ocl.cl choice).
// Mirrors the USE_2D_BLOCK_IO_K_I8 dequant path: 8b VNNI-transform read -> per-byte
// dequant -> (half) cast into the DPAS A operand k_raw (lane=head, elem=key; NOT VNNI).
// Compare against k_dequant_half.cl via compare_k_dequant.sh. Float wins (fewer movs)
// because GEN forces the int-word->half conversion mov to a scattered <2> dst, which the
// half path then has to repack; float stays dense <1>. See sdpa-ocl-int8-perf memory.
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#define SUBGROUP_SIZE 16
#define KEY_ZERO_POINTS 1
#define kq_sg_tile_keys 16
#define kq_key_blocks 2
#define DPAS_K 16
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void t(const global void* K,int KD_w,int KD_h,int KD_p,int db,int key_base,
                global float* Ks, global float* Kz, global int* out){
    int lane=get_sub_group_local_id();
    float k_scale_lane[kq_sg_tile_keys/SUBGROUP_SIZE];
    float k_zp_lane[kq_sg_tile_keys/SUBGROUP_SIZE];
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
            const float k_sc=sub_group_broadcast(k_scale_lane[krel/SUBGROUP_SIZE],krel%SUBGROUP_SIZE);
            const float k_zp=sub_group_broadcast(k_zp_lane[krel/SUBGROUP_SIZE],krel%SUBGROUP_SIZE);
            const float deq_k=(convert_float(raw4[bb])-k_zp)*k_sc;
            k_raw[krel/8][krel%8]=as_ushort((half)deq_k);
        }
    }
    ((global int8*)out)[lane]=as_int8((ushort16)(k_raw[0],k_raw[1]));
}
