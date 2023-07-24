// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>

#include <cpu/aarch64/jit_generator.hpp>
#include <ie_common.h>

#include "snippets/snippets_isa.hpp"
#include "snippets/generator.hpp"
#include "node.h"


namespace ov {
namespace intel_cpu {
namespace aarch64 {

enum emitter_in_out_map {
    vec_to_vec,
    vec_to_gpr,
    gpr_to_vec,
    gpr_to_gpr,
};

// structure for storage of emitter parameters to hash in map
struct emitter_params {
    virtual size_t hash() const = 0;
};

class jit_emitter : public ov::snippets::Emitter {
public:
    jit_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                const float alpha = 0.f,
                emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_vec) :
                Emitter(nullptr), h(host), host_isa_(host_isa), exec_prc_(exec_prc), alpha(alpha), in_out_type_(in_out_type) {
    }

    jit_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                const std::shared_ptr<ngraph::Node>& n,
                InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32,
                const float alpha = 0.f,
                emitter_in_out_map in_out_type = emitter_in_out_map::vec_to_vec) :
                Emitter(n), h(host), host_isa_(host_isa), exec_prc_(exec_prc), alpha(alpha), in_out_type_(in_out_type) {
    }

    void emit_code(
        const std::vector<size_t> &in_idxs,
        const std::vector<size_t> &out_idxs,
        const std::vector<size_t> &pool_vec_idxs = {},
        const std::vector<size_t> &pool_gpr_idxs = {}) const override;

    virtual size_t get_inputs_count() const = 0;
    virtual size_t get_aux_vecs_count() const;
    virtual size_t get_aux_gprs_count() const;

    /**
     * @brief Returns supported precisions.
     * Precisions are ordered, the first bigger bitness precision with the same type will be selected.
     * Empty collection means the emitter supports any input precisions.
     */
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ngraph::Node>& node = nullptr);

protected:
    size_t get_max_vecs_count() const;
    size_t get_vec_length() const;

    mutable std::vector<uint32_t> aux_vec_idxs;
    mutable std::vector<uint32_t> aux_gpr_idxs;

    dnnl::impl::cpu::aarch64::jit_generator* h;
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa_;
    InferenceEngine::Precision exec_prc_;
    const float alpha;

    emitter_in_out_map in_out_type_;

    virtual void prepare_table();
    virtual void register_table_entries() {}

    virtual void emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const = 0;

    virtual void emitter_preamble(const std::vector<size_t>& in_idxs,
                                  const std::vector<size_t>& out_idxs,
                                  const std::vector<size_t>& pool_aux_vec_idxs,
                                  const std::vector<size_t>& pool_aux_gpr_idxs) const;

    virtual void emitter_postamble() const;

private:
    mutable std::vector<size_t> preserved_vec_idxs;
    mutable std::vector<size_t> preserved_gpr_idxs;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
