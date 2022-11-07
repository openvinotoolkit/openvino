// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <math.h>
#include <dnnl_types.h>
#include "dnnl_extension_utils.h"
#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include "ie_parallel.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/kernels/jit_kernel_base.hpp"
#include "nodes/kernels/jit_kernel_traits.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;
using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::cpu::x64::eltwise_injector;

struct jit_refine_anchors_conf {
};

struct jit_refine_anchors_call_args {
    const float* deltas;
    const float* scores;
    const float* anchors;
    float* proposals;
    const int32_t h;
    const int32_t w;
    const int32_t anchors_num;
    const int32_t* refine_anchor_indices;
    const uint32_t* refine_anchor_masks;
    uint32_t anchor_start_idx;
    uint32_t anchor_anchor_offset;
    uint32_t anchor_idx_offset;
    uint32_t delta_start_idx;
    uint32_t delta_anchor_offset;
    uint32_t delta_idx_offset;
    uint32_t score_start_idx;
    uint32_t score_anchor_offset;
    uint32_t proposal_start_idx;
    uint32_t proposal_anchor_offset;
    uint32_t proposal_idx_offset;
    const float img_h;
    const float img_w;
    const float min_box_h;
    const float min_box_w;
    const float max_delta_log_wh;
    const float coordinates_offset;
};

using jit_refine_anchors_kernel = jit_kernel_tbase<jit_refine_anchors_conf, jit_refine_anchors_call_args>;

template <x64::cpu_isa_t isa>
class jit_refine_anchors_kernel_fp32 : public jit_refine_anchors_kernel {
 public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_refine_anchors_kernel_fp32)

    using Vmm = typename jit_kernel_traits<isa, ov::element::Type_t::f32>::Vmm;
    static constexpr unsigned VCMPPS_LE = jit_kernel_traits<isa, ov::element::Type_t::f32>::VCMPPS_LE;
    static constexpr unsigned VCMPPS_LT = jit_kernel_traits<isa, ov::element::Type_t::f32>::VCMPPS_LT;
    static constexpr unsigned VCMPPS_GT = jit_kernel_traits<isa, ov::element::Type_t::f32>::VCMPPS_GT;
    static constexpr unsigned SIMD_WIDTH = jit_kernel_traits<isa, ov::element::Type_t::f32>::SIMD_WIDTH;

    jit_refine_anchors_kernel_fp32(const jit_refine_anchors_conf &jqp)
        : jit_refine_anchors_kernel(isa, jqp) {}

    void generate() override {
        jit_refine_anchors_kernel::generate();
        exp_injector->prepare_table();
    }

    void generate(RegistersPool::Ptr registers_pool, StackAllocator::Ptr stack_allocator) override;

 private:
    void update_input_output_ptrs();

    template<typename Tmm>
    void uni_expf(const Tmm& arg) {
        exp_injector->compute_vector(arg.getIdx());
    }

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector =
        std::make_shared<jit_uni_eltwise_injector_f32<isa>>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f);

    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_anchors_loop = rcx;

    // Stable variables
    Xbyak::Reg64 reg_anchors_ptr = r8;
    Xbyak::Reg64 reg_deltas_ptr = r9;
    Xbyak::Reg64 reg_scores_ptr = r10;
    Xbyak::Reg64 reg_proposals_ptr = r11;
    Xbyak::Reg64 reg_anchors_chunk = r12;
    Xbyak::Reg64 reg_img_h = r13;
    Xbyak::Reg64 reg_img_w = r14;
    Xbyak::Reg64 reg_num_proc_elem = r15;

    Vmm vmm_x0 = Vmm(0);
    Vmm vmm_y0 = Vmm(1);
    Vmm vmm_x1 = Vmm(2);
    Vmm vmm_y1 = Vmm(3);
    Vmm vmm_dx = Vmm(4);
    Vmm vmm_dy = Vmm(5);
    Vmm vmm_d_log_w = Vmm(6);
    Vmm vmm_d_log_h = Vmm(7);
};

} // namespace intel_cpu
} // namespace ov
