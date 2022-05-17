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

#include "gpu/ocl/ocl_math_utils.h"
#pragma OPENCL_EXTENSION cl_intel_subgroups_short : enable

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 16, 1))) __kernel void
xe_hp_wei_convert_f32_to_bf16(const __global float *wei_f32,
        const __global float *bia_f32, __global ushort *wei_bf16,
        __global ushort *bia_bf16) {

    const int gid[2] = {get_group_id(0), get_group_id(1)};
    const int sg_id = get_sub_group_id();
    const int sg_loc_id = get_sub_group_local_id();
    const int kw = (gid[0] % KW) * IC_BLOCK * OC_BLOCK;
    const int kh = ((gid[0] / KW) % KH) * KW * IC_BLOCK * OC_BLOCK;
    const int kd = ((gid[0] / KH / KW) % KD) * KH * KW * IC_BLOCK * OC_BLOCK;
    const int ic = (gid[0] / KD / KH / KW) * KD * KH * KW * IC_BLOCK * OC_BLOCK;
#if WEI_DT_BF16
    const int oc = (gid[1] % (OC / LWS_1)) * IC * KD * KH * KW * OC_BLOCK
            + sg_id * IC_BLOCK;
    const int g = (gid[1] / (OC / LWS_1)) * OC * IC * KD * KH * KW;
    float2 tmp = as_float2(intel_sub_group_block_read2(
            (const __global uint *)(wei_f32 + g + oc + ic + kd + kh + kw)));
    intel_sub_group_block_write_us2(
            wei_bf16 + g + oc + ic + kd + kh + kw, cvt_f32_to_bf16(tmp));
#endif

#if WITH_BIAS && BIA_DT_BF16
    if (gid[0] == 0 && sg_loc_id == 0) {
        const int bia_off = gid[1] / (OC / OC_BLOCK) * OC
                + (gid[1] % (OC / OC_BLOCK)) * OC_BLOCK + sg_id;
        bia_bf16[bia_off] = cvt_f32_to_bf16(bia_f32[bia_off]);
    }
#endif
}
