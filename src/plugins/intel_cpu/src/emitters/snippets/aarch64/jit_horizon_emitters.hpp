// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <memory>
#include <set>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_horizon_max_emitter : public jit_emitter {
public:
    jit_horizon_max_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);

    jit_horizon_max_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

class jit_horizon_sum_emitter : public jit_emitter {
public:
    jit_horizon_sum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            ov::element::Type exec_prc = ov::element::f32);

    jit_horizon_sum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                            dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                            const std::shared_ptr<ov::Node>& node);

    size_t get_inputs_count() const override;

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const override;

    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const;
};

}  // namespace ov::intel_cpu::aarch64
