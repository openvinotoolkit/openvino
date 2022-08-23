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
    int32_t anchors_chunk;
};

struct jit_refine_anchors_call_args {
    const float* deltas;
    const float* scores;
    const float* anchors;
    float* proposals;
    const int32_t h;
    const int32_t w;
    const int32_t anchors_num;
    uint32_t anchor_idx_offset;
    uint32_t anchor_chunk_offset;
    uint32_t delta_idx_offset;
    uint32_t delta_chunk_offset;
    uint32_t score_idx_offset;
    uint32_t score_chunk_offset;
    uint32_t proposal_idx_offset;
    uint32_t proposal_chunk_offset;
    const float img_h;
    const float img_w;
    const float min_box_h;
    const float min_box_w;
    const float max_delta_log_wh;
    const float coordinates_offset;
};

using jit_refine_anchors_kernel = jit_kernel_base<jit_refine_anchors_conf, jit_refine_anchors_call_args>;

template <x64::cpu_isa_t isa>
class jit_refine_anchors_kernel_fp32 : public jit_refine_anchors_kernel {
 public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_refine_anchors_kernel_fp32)

    static constexpr auto KERNEL_ELEMENT_TYPE = ov::element::Type_t::f32;

    using Vmm = typename jit_kernel_traits<isa, KERNEL_ELEMENT_TYPE>::Vmm;
    static constexpr unsigned SIMD_WIDTH = jit_kernel_traits<isa, KERNEL_ELEMENT_TYPE>::SIMD_WIDTH;
    static constexpr unsigned XMM_SIMD_WIDTH = 16 / sizeof(typename ov::element_type_traits<KERNEL_ELEMENT_TYPE>::value_type);
    static constexpr unsigned YMM_SIMD_WIDTH = 32 / sizeof(typename ov::element_type_traits<KERNEL_ELEMENT_TYPE>::value_type);
    static constexpr unsigned ZMM_SIMD_WIDTH = 64 / sizeof(typename ov::element_type_traits<KERNEL_ELEMENT_TYPE>::value_type);
    static constexpr unsigned DTYPE_SIZE = sizeof(typename ov::element_type_traits<KERNEL_ELEMENT_TYPE>::value_type);

    jit_refine_anchors_kernel_fp32(const jit_refine_anchors_conf &jqp)
        : jit_refine_anchors_kernel(isa, jqp) {}

    void generate() override;

 private:
    static constexpr uint8_t mask[] = {
        0xFF, 0xFF, 0xFF, 0xFF,
    };

    void update_input_output_ptrs();

    inline void push_xmm(const Xbyak::Xmm &xmm) {
        size_t simd_width = 0;
        if (xmm.isXMM()) {
            simd_width = 4;
        } else if (xmm.isYMM()) {
            simd_width = 8;
        } else {
            simd_width = 16;
        }
        sub(rsp, simd_width);
        uni_vmovdqu(ptr[rsp], xmm);
    }

    inline void pop_xmm(const Xbyak::Xmm &xmm) {
        size_t simd_width = 0;
        if (xmm.isXMM()) {
            simd_width = 4;
        } else if (xmm.isYMM()) {
            simd_width = 8;
        } else {
            simd_width = 16;
        }
        uni_vmovdqu(xmm, ptr[rsp]);
        add(rsp, simd_width);
    }

    inline void emulate_gather(const Xbyak::Xmm &xmm_arg,
                               const Xbyak::Reg64 &num_elem,
                               const Xbyak::Reg64 &mem_base,
                               const Xbyak::Reg64 &idx_offset,
                               const Xbyak::Reg64 &anchor_idx) {
        int simd_width = 0;
        if (!xmm_arg.isXMM()) {
            simd_width = SIMD_WIDTH;
        } else if (xmm_arg.isYMM()) {
            simd_width = YMM_SIMD_WIDTH;
        } else {
            simd_width = ZMM_SIMD_WIDTH;
        }

        std::vector<int> xmm_idxs{
            0, 1, 2, 3, 4, 5, 6, 7
        };
        xmm_idxs.erase(std::remove(xmm_idxs.begin(), xmm_idxs.end(), xmm_arg.getIdx()), xmm_idxs.end());
        Xbyak::Xmm temp{xmm_idxs.front()};

        std::vector<int> reg_idxs{
            0,  1,  2,  3,  5,  6,  7,
            8,  9, 10, 11, 12, 13, 14, 15
        };
        reg_idxs.erase(std::remove_if(reg_idxs.begin(), reg_idxs.end(),
            [&](const int& idx) {
                return idx == num_elem.getIdx() ||
                       idx == mem_base.getIdx() ||
                       idx == idx_offset.getIdx() ||
                       idx == anchor_idx.getIdx();
            }), reg_idxs.end());
        Xbyak::Reg64 idx{reg_idxs.front()};
        push(idx);
        push_xmm(temp);
        xor_(idx, idx);
        Xbyak::Label gather_end;

        size_t i_elem = 0;
        for (int i = 0; i < simd_width/XMM_SIMD_WIDTH; i++) {
            for (int j = 0; j < XMM_SIMD_WIDTH; j++) {
                cmp(num_elem, i_elem);
                jbe(gather_end, T_NEAR);
                Xbyak::Address addr = ptr[mem_base + anchor_idx * DTYPE_SIZE];
                switch (DTYPE_SIZE) {
                    case 4: uni_vpinsrd(temp, temp, addr, j); break;
                    case 2: uni_vpinsrw(temp, temp, addr, j); break;
                    case 1: uni_vpinsrb(temp, temp, addr, j); break;
                    default: IE_THROW() << "The data type of size '" << DTYPE_SIZE << "' is not supported.";
                }
                add(anchor_idx.cvt32(), idx_offset.cvt32());
                i_elem += 1;
            }
            if (mayiuse(cpu_isa_t::avx512_core)) {
                vinsertf32x4(Xbyak::Zmm{xmm_arg.getIdx()}, Xbyak::Zmm{xmm_arg.getIdx()}, temp, i);
            } else if (mayiuse(cpu_isa_t::avx2)) {
                vinsertf128(Xbyak::Ymm{xmm_arg.getIdx()}, Xbyak::Ymm{xmm_arg.getIdx()}, temp, i);
            }
        }
        L(gather_end);
        pop_xmm(temp);
        pop(idx);
    }

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

    // Temp variables
    Xbyak::Reg64 reg_anchor_idx_offset = r13;
    Xbyak::Reg64 reg_anchor_chunk_offset = r14;
    Xbyak::Reg64 reg_delta_idx_offset = r13;
    Xbyak::Reg64 reg_delta_chunk_offset = r14;
    Xbyak::Reg64 reg_score_idx_offset = r13;
    Xbyak::Reg64 reg_score_chunk_offset = r14;
    Xbyak::Reg64 reg_proposal_idx_offset = r13;
    Xbyak::Reg64 reg_proposal_chunk_offset = r14;
    Xbyak::Reg64 reg_anchor_idx = r15;
    Xbyak::Reg64 reg_num_proc_elem = r13;
    Xbyak::Reg64 reg_elem_size = r14;

    Vmm vmm_x0 = Vmm(0);
    Vmm vmm_y0 = Vmm(1);
    Vmm vmm_x1 = Vmm(2);
    Vmm vmm_y1 = Vmm(3);
    Vmm vmm_dx = Vmm(4);
    Vmm vmm_dy = Vmm(5);
    Vmm vmm_d_log_w = Vmm(6);
    Vmm vmm_d_log_h = Vmm(7);
    Vmm vmm_score = Vmm(0);
};

//template <x64::cpu_isa_t isa>
//template <>
//struct jit_refine_anchors_kernel_fp32<isa>::jit_traits<x64::sse41> {
//    using Vmm = Xbyak::Xmm;
//    static constexpr unsigned SIMD_WIDTH = x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(typename ov::element_type_traits<KERNEL_ELEMENT_TYPE>::value_type);
//};

}
}
