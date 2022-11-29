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
#include "nodes/kernels/jit_kernel.hpp"

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

using jit_refine_anchors_kernel = jit_kernel<jit_refine_anchors_conf, jit_refine_anchors_call_args>;

template <x64::cpu_isa_t isa>
class jit_refine_anchors_kernel_fp32 : public jit_refine_anchors_kernel {
 public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_refine_anchors_kernel_fp32)

    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    static constexpr unsigned VCMPPS_LE = 0x02;
    static constexpr unsigned VCMPPS_LT = 0x05;
    static constexpr unsigned VCMPPS_GT = 0x0e;
    static constexpr unsigned SIMD_WIDTH = x64::cpu_isa_traits<isa>::vlen / sizeof(typename ov::element_type_traits<ov::element::Type_t::f32>::value_type);

    jit_refine_anchors_kernel_fp32(const jit_refine_anchors_conf &jqp)
        : jit_refine_anchors_kernel(jit_name(), isa, jqp) {}

    void generate() override {
        jit_refine_anchors_kernel::generate();
        exp_injector->prepare_table();
    }

    void generate_impl() override;

 protected:
    void createStackAllocator() override {
        stackAllocator = std::unique_ptr<StackAllocator>(new StackAllocator{*this, 0x40});
    }

 private:
    void update_input_output_ptrs();

    inline void gather1(Vmm dst,
                        Xbyak::Reg64 src,
                        Vmm idx,
                        int k_mask_idx,
                        const StackAllocator::RegAddress<Vmm>& vmm_mask);

    inline void gather4(std::array<Vmm, 4> dst,
                        Xbyak::Reg64 src,
                        Vmm idx,
                        Vmm idx_offset,
                        int k_mask_idx,
                        const StackAllocator::RegAddress<Vmm>& vmm_mask);

    inline void scatter1(Xbyak::Reg64 dst,
                         Vmm idx,
                         Vmm src,
                         int k_mask_idx,
                         const StackAllocator::RegAddress<Vmm>& vmm_mask);

    inline void scatter4(Xbyak::Reg64 dst,
                         Vmm idx,
                         std::array<Vmm, 4> src,
                         Vmm idx_offset,
                         int k_mask_idx,
                         const StackAllocator::RegAddress<Vmm>& vmm_mask);

    template<typename Tmm>
    void uni_expf(const Tmm& arg) {
        exp_injector->compute_vector(arg.getIdx());
    }

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector =
        std::make_shared<jit_uni_eltwise_injector_f32<isa>>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f);

    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_anchors_loop = abi_not_param1;

    // Stable variables
    Xbyak::Reg64 reg_anchors_ptr = r8;
    Xbyak::Reg64 reg_deltas_ptr = r9;
    Xbyak::Reg64 reg_scores_ptr = r10;
    Xbyak::Reg64 reg_proposals_ptr = r11;
    Xbyak::Reg64 reg_anchors_chunk = r12;
    Xbyak::Reg64 reg_img_h = r13;
    Xbyak::Reg64 reg_img_w = r14;
    Xbyak::Reg64 reg_num_proc_elem = r15;
};

} // namespace intel_cpu
} // namespace ov
