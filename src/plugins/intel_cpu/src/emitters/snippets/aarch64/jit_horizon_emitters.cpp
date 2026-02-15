// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_horizon_emitters.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::aarch64 {

/// horizon_max ///
jit_horizon_max_emitter::jit_horizon_max_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_horizon_max_emitter::jit_horizon_max_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                 const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_horizon_max_emitter::get_inputs_count() const {
    return 1;
}

void jit_horizon_max_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                        const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_horizon_max_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                       const std::vector<size_t>& out_vec_idxs) const {
    h->fmaxv(Xbyak_aarch64::SReg(out_vec_idxs[0]), Xbyak_aarch64::VReg4S(in_vec_idxs[0]));
}

std::set<std::vector<element::Type>> jit_horizon_max_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// horizon_sum ///
jit_horizon_sum_emitter::jit_horizon_sum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_horizon_sum_emitter::jit_horizon_sum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                 const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_horizon_sum_emitter::get_inputs_count() const {
    return 1;
}

void jit_horizon_sum_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                        const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_horizon_sum_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                       const std::vector<size_t>& out_vec_idxs) const {
    const uint32_t in = in_vec_idxs[0];
    const uint32_t out = out_vec_idxs[0];

    h->faddp(Xbyak_aarch64::VReg4S(out), Xbyak_aarch64::VReg4S(in), Xbyak_aarch64::VReg4S(in));
    h->faddp(Xbyak_aarch64::VReg2S(out), Xbyak_aarch64::VReg2S(out), Xbyak_aarch64::VReg2S(out));
}

std::set<std::vector<element::Type>> jit_horizon_sum_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

}  // namespace ov::intel_cpu::aarch64
