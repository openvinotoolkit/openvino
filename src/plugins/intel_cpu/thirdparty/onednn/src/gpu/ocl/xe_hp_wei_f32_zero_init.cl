/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

void __builtin_IB_simd_block_write_1_global_l(__global ulong *, ulong);

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 16, 1))) __kernel void
xe_hp_wei_f32_zero_init(__global float *wei_f32, __global float *bia_f32) {
    const int gid[2] = {get_group_id(0), get_group_id(1)};
    const int sg_id = get_sub_group_id();
    const int sg_loc_id = get_sub_group_local_id();
    const int kw = (gid[0] % KW) * IC_BLOCK * OC_BLOCK;
    const int kh = ((gid[0] / KW) % KH) * KW * IC_BLOCK * OC_BLOCK;
    const int kd = ((gid[0] / KH / KW) % KD) * KH * KW * IC_BLOCK * OC_BLOCK;
    const int ic = (gid[0] / KD / KH / KW) * KD * KH * KW * IC_BLOCK * OC_BLOCK;
    const int oc = (gid[1] % (OC / LWS_1)) * IC * KD * KH * KW * OC_BLOCK
            + sg_id * IC_BLOCK;
    const int g = (gid[1] / (OC / LWS_1)) * OC * IC * KD * KH * KW;
    long zero = 0;
    __builtin_IB_simd_block_write_1_global_l(
            (__global ulong *)(wei_f32 + g + oc + ic + kd + kh + kw), zero);
#if WITH_BIAS
    if (gid[0] == 0 && sg_loc_id == 0) {
        const int bia_off = gid[1] * OC_BLOCK + sg_id;
        bia_f32[bia_off] = 0.0f;
    }
#endif
}
