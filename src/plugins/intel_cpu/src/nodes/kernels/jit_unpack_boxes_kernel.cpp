// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_unpack_boxes_kernel.hpp"

namespace ov {
namespace intel_cpu {

template <x64::cpu_isa_t isa>
void jit_unpack_boxes_kernel_fp32<isa>::generate() {
    this->preamble();

    xor_(reg_box_idx, reg_box_idx);
    xor_(reg_count, reg_count);
    xor_(reg_i, reg_i);
    xor_(reg_pre_nms_topn, reg_pre_nms_topn);
    mov(reg_i.cvt32(), ptr[reg_params + offsetof(jit_unpack_boxes_call_args, i)]);
    mov(reg_index_ptr, ptr[reg_params + offsetof(jit_unpack_boxes_call_args, index)]);
    mov(reg_pre_nms_topn.cvt32(), ptr[reg_params + offsetof(jit_unpack_boxes_call_args, pre_nms_topn)]);
    mov(reg_p_proposals_ptr, ptr[reg_params + offsetof(jit_unpack_boxes_call_args, p_proposals)]);
    mov(reg_unpacked_boxes_ptr, ptr[reg_params + offsetof(jit_unpack_boxes_call_args, unpacked_boxes)]);
    mov(reg_is_dead_ptr, ptr[reg_params + offsetof(jit_unpack_boxes_call_args, is_dead)]);

    /*
     * @see
        unpacked_boxes[0*pre_nms_topn + i] = p_proposals[6*i + 0];
        unpacked_boxes[1*pre_nms_topn + i] = p_proposals[6*i + 1];
        unpacked_boxes[2*pre_nms_topn + i] = p_proposals[6*i + 2];
        unpacked_boxes[3*pre_nms_topn + i] = p_proposals[6*i + 3];
        unpacked_boxes[4*pre_nms_topn + i] = p_proposals[6*i + 4];
        is_dead[i] = (p_proposals[6*i + 5] == 1.0) ? 0 : 1;
     */

    uni_vmovdqu(xmm_index, ptr[reg_index_ptr]);
    uni_vpcmpeqd(xmm_mask, xmm_mask, xmm_mask);
    vgatherdps(xmm_proposals, ptr[reg_p_proposals_ptr + xmm_index], xmm_mask);

    this->postamble();
}

template struct jit_unpack_boxes_kernel_fp32<x64::avx512_core>;
template struct jit_unpack_boxes_kernel_fp32<x64::avx2>;
template struct jit_unpack_boxes_kernel_fp32<x64::sse41>;

}
}
