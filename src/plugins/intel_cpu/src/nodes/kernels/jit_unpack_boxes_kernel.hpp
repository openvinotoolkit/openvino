// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <math.h>
#include <dnnl_types.h>
#include <dnnl_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "ie_parallel.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "jit_kernel_base.hpp"
#include "jit_kernel_traits.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;
using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

struct jit_unpack_boxes_conf {
};

struct jit_unpack_boxes_call_args {
    int32_t i;
    int32_t* index;
    int32_t pre_nms_topn;
    int32_t* is_dead;
    const float* p_proposals;
    float* unpacked_boxes;
};

using jit_unpack_boxes_kernel = jit_kernel_base<jit_unpack_boxes_conf, jit_unpack_boxes_call_args>;

template <x64::cpu_isa_t ISA>
class jit_unpack_boxes_kernel_fp32 : public jit_unpack_boxes_kernel {
 public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_unpack_boxes_kernel_fp32)

    static constexpr auto KERNEL_ELEMENT_TYPE = ov::element::Type_t::f32;
    using Vmm = typename jit_kernel_traits<ISA, KERNEL_ELEMENT_TYPE>::Vmm;

    using jit_unpack_boxes_kernel::jit_unpack_boxes_kernel;

    void generate() override;

 private:
    static constexpr unsigned VCMPPS_LE = 0x02;
    static constexpr unsigned VCMPPS_GT = 0x0e;
    static constexpr unsigned SIMD_WIDTH = jit_kernel_traits<ISA, KERNEL_ELEMENT_TYPE>::SIMD_WIDTH;

    static constexpr uint8_t mask[] = {
        0xFF, 0xFF, 0xFF, 0xFF,
    };

    Xmm xmm_index = Xmm(0);
    Xmm xmm_mask = Xmm(1);
    Xmm xmm_proposals = Xmm(2);

    Xbyak::Reg64 reg_box_idx = r8;
    Xbyak::Reg64 reg_count = r9;
    Xbyak::Reg64 reg_tail = r10;
    Xbyak::Reg64 reg_i = r11;
    Xbyak::Reg64 reg_index_ptr = r12;
    Xbyak::Reg64 reg_pre_nms_topn = r13;
    Xbyak::Reg64 reg_p_proposals_ptr = r14;
    Xbyak::Reg64 reg_unpacked_boxes_ptr = r15;
    Xbyak::Reg64 reg_is_dead_ptr = rcx;
    Xbyak::Reg64 reg_params = abi_param1;
};

}
}
