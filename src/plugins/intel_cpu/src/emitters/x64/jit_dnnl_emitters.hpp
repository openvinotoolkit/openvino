// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>
#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {

class jit_dnnl_emitter : public jit_emitter {
public:
    void emit_code(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const override;

    void emit_data() const override;

    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const override {};

    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

    void print_debug_info() const override;

protected:
    jit_dnnl_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                       dnnl_alg_kind_t algKind, float inpAlpha, float inpBeta,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_dnnl_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    void set_injector();

    dnnl_alg_kind_t kind {dnnl_alg_kind_undef};
    float alpha {0.f};
    float beta {0.f};

    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<dnnl::impl::cpu::x64::sse41>> eltwise_injector_sse42;
    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<dnnl::impl::cpu::x64::avx2>> eltwise_injector_avx2;
    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector_f32<dnnl::impl::cpu::x64::avx512_core>> eltwise_injector_avx512_core;

private:
    size_t get_inputs_num() const override;
};

class jit_dnnl_aux_emitter : public jit_dnnl_emitter {
public:
    jit_dnnl_aux_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                           dnnl_alg_kind_t algKind, float inpAlpha, float inpBeta,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    void print_debug_info() const override;

private:
};

}   // namespace intel_cpu
}   // namespace ov
