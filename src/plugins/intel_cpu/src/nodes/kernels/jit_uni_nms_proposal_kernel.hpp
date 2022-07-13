// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cpu/x64/jit_generator.hpp>
#include <cassert>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {
namespace details {
namespace x64 = dnnl::impl::cpu::x64;
} // namespace details

class jit_uni_nms_proposal_kernel {
public:
    struct jit_nms_conf {
        int max_num_out;
        float nms_threshold;
        float coordinates_offset;
    };

    struct jit_nms_call_args {
        int pre_nms_topn;
        int *is_dead;
        float *x0;
        float *x1;
        float *y0;
        float *y1;
        int *index_out;
        std::size_t *const num_out;
    };

    void operator()(const jit_nms_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_nms_proposal_kernel(jit_nms_conf jcp) :
        jcp_(jcp) {
    }
    virtual ~jit_uni_nms_proposal_kernel() = default;

    virtual void create_ker() = 0;

protected:
    void (*ker_)(const jit_nms_call_args*) = nullptr;
    jit_nms_conf jcp_;
};

template <details::x64::cpu_isa_t isa>
class jit_uni_nms_proposal_kernel_impl : public jit_uni_nms_proposal_kernel, public details::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_nms_proposal_kernel_impl)

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    static constexpr unsigned VCMPPS_LE = 0x02;
    static constexpr unsigned VCMPPS_GT = 0x0e;
    static constexpr unsigned VCMPPS_GE = 0x05;
    static constexpr unsigned VCMPPS_ORD = 0x07;

    explicit jit_uni_nms_proposal_kernel_impl(const jit_nms_conf& jcp);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == details::x64::sse41,
        Xbyak::Xmm, isa == details::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    void generate() override;

protected:
    // TODO: Move to jit_kernel_base
    void uni_vsubss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2);

    // TODO: Move to jit_kernel_base
    void uni_vmulss(const Xbyak::Xmm &x, const Xbyak::Operand &op1,
                    const Xbyak::Operand &op2);

private:
    static constexpr unsigned simd_width = details::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    Xbyak::Reg64 reg_box_idx = r8;
    Xbyak::Reg64 reg_count = r9;
    Xbyak::Reg64 reg_tail = r10;
    Xbyak::Reg64 reg_pre_nms_topn = r11;
    Xbyak::Reg64 reg_x0_ptr = r12;
    Xbyak::Reg64 reg_x1_ptr = r13;
    Xbyak::Reg64 reg_y0_ptr = r14;
    Xbyak::Reg64 reg_y1_ptr = r15;
    Xbyak::Reg64 reg_simd_tail_len = rdx;
    Xbyak::Reg64 reg_is_dead_ptr;
    Xbyak::Reg64 reg_params;
};

} // namespace node
} // namespace intel_cpu
} // namespace ov
