// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/jit_generator.hpp>
#include "jit_emitter.hpp"
#include "jit_bf16_emitters.hpp"

namespace ov {
namespace intel_cpu {

class jit_convert_emitter : public jit_emitter {
public:
    jit_convert_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                        InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out,
                   const std::vector<size_t>& pool, const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override;

    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    void emit_data() const override;

    void float2bfloat(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void dword2sint8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;
    template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
    void dword2uint8(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const;

    ov::element::Type input_type;
    ov::element::Type output_type;

    const ov::element::TypeVector supported_types = {
            ov::element::f32,
            ov::element::bf16,
            ov::element::i8,
            ov::element::u8
    };

    std::shared_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16 = nullptr;
};

}   // namespace intel_cpu
}   // namespace ov
