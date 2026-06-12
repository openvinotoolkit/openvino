// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <functional>

namespace ov::intel_cpu {

struct weights_decompression_compile_params_t {
    bool with_scales;
    bool with_zero_points;
    bool broadcast_scales;
    bool broadcast_zero_points;
    size_t oc_size;
    size_t ic_internal_size;
    dnnl::impl::data_type_t weights_dt;
    dnnl::impl::data_type_t decomp_buffer_dt;
    dnnl::impl::data_type_t scales_dt;
    dnnl::impl::data_type_t zero_points_dt;
};

struct weights_decompression_runtime_params_t {
    const void* weights_ptr;
    const void* decomp_buffer_ptr;
    const void* scales_ptr;
    const void* zero_points_ptr;
    size_t ic_size;
};

struct jit_weights_decompression_kernel_base_t {
    void operator()(const weights_decompression_runtime_params_t* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_weights_decompression_kernel_base_t(const weights_decompression_compile_params_t& jcp) : jcp_(jcp) {}
    virtual ~jit_weights_decompression_kernel_base_t() = default;

protected:
    void (*ker_)(const weights_decompression_runtime_params_t*) = nullptr;
    weights_decompression_compile_params_t jcp_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_brgemm_weights_decompression_kernel_t : public jit_weights_decompression_kernel_base_t,
                                                   public dnnl::impl::cpu::x64::jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_weights_decompression_kernel_t)

    explicit jit_brgemm_weights_decompression_kernel_t(const weights_decompression_compile_params_t& jcp);

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    static constexpr int n_vregs = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::n_vregs;

    void generate() override;
    void init_decomp_params(std::function<Vmm(int)> vmm_params,
                            Xbyak::Reg64 reg_params,
                            bool broadcast_values,
                            dnnl::impl::data_type_t element_type);
    void load_weights(Vmm vmm_load, const Xbyak::Address& addr, int ic);
    void store_weights(const Xbyak::Address& addr, Vmm vmm_store);

    Vmm vmm_scales(int ocb) {
        return Vmm(unroll_factor + ocb);
    }
    Vmm vmm_zero_points(int ocb) {
        return Vmm(2 * unroll_factor + ocb);
    }
    Vmm vmm_weights(int ocb) {
        assert(ocb < unroll_factor);
        return Vmm(ocb);
    }

    Vmm vmm_tmp(int idx) {
        return Vmm(n_vregs - idx - 1);
    }

    Vmm vmm_lookup() {
        return vmm_tmp(0);
    }
    Vmm vmm_lookup_low() {
        return vmm_tmp(0);
    }
    Vmm vmm_lookup_high() {
        return vmm_tmp(1);
    }
    Vmm vmm_mask() {
        return vmm_tmp(1);
    }
    Vmm vmm_mask8() {
        return vmm_tmp(2);
    }
    Vmm vmm_mask7() {
        return vmm_tmp(3);
    }
    Vmm vmm_aux0() {
        return Vmm(14);
    }
    Vmm vmm_aux1() {
        return Vmm(15);
    }

    Xbyak::Reg64 reg_weights = Xbyak::Reg64(8);
    Xbyak::Reg64 reg_decomp_buffer = Xbyak::Reg64(9);
    Xbyak::Reg64 reg_scales = Xbyak::Reg64(10);
    Xbyak::Reg64 reg_zero_points = Xbyak::Reg64(11);
    Xbyak::Reg64 reg_ic_size = Xbyak::Reg64(12);
    Xbyak::Reg64 reg_tmp = Xbyak::Reg64(13);

    size_t vec_size;
    static constexpr int unroll_factor = 4;
};

}  // namespace ov::intel_cpu
