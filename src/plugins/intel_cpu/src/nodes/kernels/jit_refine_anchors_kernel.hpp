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
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
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
using namespace dnnl::impl::cpu::x64::eltwise_injector;

struct jit_refine_anchors_conf {
};

struct jit_refine_anchors_call_args {
    int32_t i;
    int32_t* index;
    int32_t pre_nms_topn;
    int32_t* is_dead;
    const float* p_proposals;
    float* unpacked_boxes;
};

using jit_refine_anchors_kernel = jit_kernel_base<jit_refine_anchors_conf, jit_refine_anchors_call_args>;

template <x64::cpu_isa_t ISA>
class jit_refine_anchors_kernel_fp32 : public jit_refine_anchors_kernel {
 public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_refine_anchors_kernel_fp32)

    static constexpr auto KERNEL_ELEMENT_TYPE = ov::element::Type_t::f32;
    using Vmm = typename jit_kernel_traits<ISA, KERNEL_ELEMENT_TYPE>::Vmm;

    using jit_refine_anchors_kernel::jit_refine_anchors_kernel;

    void generate() override;

 private:
    static constexpr unsigned VCMPPS_LE = 0x02;
    static constexpr unsigned VCMPPS_GT = 0x0e;
    static constexpr unsigned SIMD_WIDTH = jit_kernel_traits<ISA, KERNEL_ELEMENT_TYPE>::SIMD_WIDTH;

    static constexpr uint8_t mask[] = {
        0xFF, 0xFF, 0xFF, 0xFF,
    };

    void uni_expf(const Vmm& arg) {
        exp_injector->compute_vector(arg.getIdx());
    }

    std::shared_ptr<jit_uni_eltwise_injector_f32<ISA>> exp_injector =
        std::make_shared<jit_uni_eltwise_injector_f32<ISA>>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.f);

    Xbyak::Xmm xmm_index = Xbyak::Xmm(0);
    Xbyak::Xmm xmm_mask = Xbyak::Xmm(1);
    Xbyak::Xmm xmm_anchors = Xbyak::Xmm(2);
    Xbyak::Xmm xmm_deltas = Xbyak::Xmm(3);

    Xbyak::Xmm reg_x0 = Xbyak::Xmm(0);
    Xbyak::Xmm reg_y0 = Xbyak::Xmm(1);
    Xbyak::Xmm reg_x1 = Xbyak::Xmm(2);
    Xbyak::Xmm reg_ww = Xbyak::Xmm(2);
    Xbyak::Xmm reg_ctr_x = Xbyak::Xmm(2);
    Xbyak::Xmm reg_y1 = Xbyak::Xmm(3);
    Xbyak::Xmm reg_hh = Xbyak::Xmm(3);
    Xbyak::Xmm reg_ctr_y = Xbyak::Xmm(3);

    Xbyak::Xmm reg_dx = Xbyak::Xmm(4);
    Xbyak::Xmm reg_pred_ctr_x = Xbyak::Xmm(4);
    Xbyak::Xmm reg_box_w = Xbyak::Xmm(4);
    Vmm reg_pred_w = Vmm(4);
    Xbyak::Xmm reg_dy = Xbyak::Xmm(5);
    Xbyak::Xmm reg_pred_ctr_y = Xbyak::Xmm(5);
    Xbyak::Xmm reg_box_h = Xbyak::Xmm(5);
    Vmm reg_pred_h = Vmm(5);
    Xbyak::Xmm reg_d_log_w = Xbyak::Xmm(6);
    Xbyak::Xmm reg_d_log_h = Xbyak::Xmm(7);

    Xbyak::Reg64 reg_anchors_ptr = r14;
    Xbyak::Reg64 reg_coordinates_offset = r14;
    Xbyak::Reg64 reg_scale_0_5 = r14;
    Xbyak::Reg64 reg_0_0 = r14;
    Xbyak::Reg64 reg_deltas_ptr = r15;
    Xbyak::Reg64 reg_max_delta_log_wh = r15;
    Xbyak::Reg64 reg_img_W = r15;
    Xbyak::Reg64 reg_img_H = r15;
    Xbyak::Reg64 reg_params = abi_param1;
};

}
}
