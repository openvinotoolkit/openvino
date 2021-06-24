// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise_injector.hpp>
#include "jit_emitter.hpp"
#include "mkldnn_node.h"



namespace MKLDNNPlugin {

class jit_mkldnn_emitter : public jit_emitter {
public:
    void emit_code(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const override;

    void emit_data() const override;

    void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs,
                   const emitter_context *emit_context = nullptr) const override {};

protected:
    jit_mkldnn_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    jit_mkldnn_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);
    void set_injector();

    mkldnn_alg_kind_t kind {mkldnn_alg_kind_undef};
    float alpha {0.f};
    float beta {0.f};

    std::shared_ptr<mkldnn::impl::cpu::x64::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::x64::sse41>> eltwise_injector_sse42;
    std::shared_ptr<mkldnn::impl::cpu::x64::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::x64::avx2>> eltwise_injector_avx2;
    std::shared_ptr<mkldnn::impl::cpu::x64::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::x64::avx512_common>> eltwise_injector_avx512_common;

private:
    size_t get_inputs_num() const override;
};

class jit_mkldnn_aux_emitter : public jit_mkldnn_emitter {
public:
    jit_mkldnn_aux_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                           InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

private:
};

} // namespace MKLDNNPlugin