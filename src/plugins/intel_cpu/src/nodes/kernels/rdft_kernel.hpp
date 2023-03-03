// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <dnnl_types.h>

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

enum dft_type {
    real_to_complex,
    complex_to_complex,
    complex_to_real,
};

template <typename T>
size_t complex_type_size() {
    return sizeof(T) * 2;
}

struct jit_dft_args {
    const void* input;
    const void* twiddles;
    void* output;
    size_t input_size;
    size_t signal_size;
    size_t output_start;
    size_t output_end;
};


struct jit_dft_kernel {
    jit_dft_kernel(bool is_inverse, enum dft_type type) : is_inverse_(is_inverse), kernel_type_(type) {}

    void (*ker_)(const jit_dft_args*);

    void operator()(const jit_dft_args* args) {
        assert(ker_);
        ker_(args);
    }

    jit_dft_kernel() : ker_(nullptr) {}
    virtual ~jit_dft_kernel() {}

    virtual void create_ker() = 0;

    bool is_inverse_;
    enum dft_type kernel_type_;
};

template <cpu_isa_t isa>
struct jit_dft_kernel_f32 : public jit_dft_kernel, public jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_dft_kernel_f32)

        jit_dft_kernel_f32(bool is_inverse, enum dft_type type) : jit_dft_kernel(is_inverse, type), jit_generator(jit_name()) {}

        void create_ker() override {
            jit_generator::create_kernel();
            ker_ = (decltype(ker_))jit_ker();
        }

        void generate() override;

    private:
        void uni_vbroadcastsd(const Xbyak::Xmm& x, const Xbyak::Operand& op);
        void uni_vbroadcastsd(const Xbyak::Ymm& x, const Xbyak::Operand& op);

        void uni_vpermilps(const Xbyak::Xmm& x, const Xbyak::Operand& op, int8_t control);
        void uni_vpermilps(const Xbyak::Ymm& x, const Xbyak::Operand& op, int8_t control);

        void load_and_broadcast_every_other_elem(const Xbyak::Zmm& x, const Xbyak::RegExp& reg_exp, const Xbyak::Xmm& tmp);
        void load_and_broadcast_every_other_elem(const Xbyak::Ymm& x, const Xbyak::RegExp& reg_exp, const Xbyak::Xmm& tmp);
        void load_and_broadcast_every_other_elem(const Xbyak::Xmm& x, const Xbyak::RegExp& reg_exp, const Xbyak::Xmm& tmp);

        int type_size = sizeof(float);

        Xbyak::Reg8 is_signal_size_even = al;
        Xbyak::Reg64 input_ptr = rbx;
        Xbyak::Reg64 input_size = r8;
        Xbyak::Reg64 output_ptr = r9;
        Xbyak::Reg64 twiddles_ptr = r10;
        Xbyak::Reg64 signal_size = r11;
        Xbyak::Reg64 output_start = r12;
        Xbyak::Reg64 output_end = r13;
};

}   // namespace intel_cpu
}   // namespace ov
