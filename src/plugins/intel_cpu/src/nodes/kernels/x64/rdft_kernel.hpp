// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef OPENVINO_ARCH_ARM64
#    include "cpu/x64/jit_generator.hpp"
#    include "dnnl_types.h"
#endif

namespace ov::intel_cpu {

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

#ifndef OPENVINO_ARCH_ARM64
struct jit_dft_kernel {
    jit_dft_kernel(bool is_inverse, enum dft_type type) : is_inverse_(is_inverse), kernel_type_(type) {}

    void (*ker_)(const jit_dft_args*) = nullptr;

    void operator()(const jit_dft_args* args) {
        assert(ker_);
        ker_(args);
    }

    jit_dft_kernel() = default;
    virtual ~jit_dft_kernel() = default;

    virtual void create_ker() = 0;

    bool is_inverse_;
    enum dft_type kernel_type_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_dft_kernel_f32 : public jit_dft_kernel, public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_dft_kernel_f32)

    jit_dft_kernel_f32(bool is_inverse, enum dft_type type)
        : jit_dft_kernel(is_inverse, type),
          jit_generator(jit_name()) {
        constexpr int simd_size = vlen / type_size;
        perm_low_values.reserve(simd_size);
        perm_high_values.reserve(simd_size);
        for (int i = 0; i < simd_size / 2; i++) {
            perm_low_values.push_back(i);
            perm_low_values.push_back(i + simd_size);
            perm_high_values.push_back(i + simd_size / 2);
            perm_high_values.push_back(i + simd_size / 2 + simd_size);
        }
    }

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    void interleave_and_store(const Vmm& real, const Vmm& imag, const Xbyak::RegExp& reg_exp, const Vmm& tmp);

    static constexpr int type_size = sizeof(float);
    static constexpr int vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;

    Xbyak::Reg8 is_signal_size_even = al;
    Xbyak::Reg64 input_ptr = rbx;
    Xbyak::Reg64 input_size = r8;
    Xbyak::Reg64 output_ptr = r9;
    Xbyak::Reg64 twiddles_ptr = r10;
    Xbyak::Reg64 signal_size = r11;
    Xbyak::Reg64 output_start = r12;
    Xbyak::Reg64 output_end = r13;

    std::vector<int> perm_low_values;
    std::vector<int> perm_high_values;

    Vmm perm_low;
    Vmm perm_high;
};

#endif
}  // namespace ov::intel_cpu
