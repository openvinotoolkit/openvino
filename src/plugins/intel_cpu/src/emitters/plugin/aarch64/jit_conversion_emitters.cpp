// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conversion_emitters.hpp"

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "emitters/plugin/aarch64/jit_conversion_helpers.hpp"
#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu::aarch64 {

jit_convert_emitter::jit_convert_emitter(jit_generator_t* host,
                                         cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node,
                                         ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      input_type(node->get_input_element_type(0)),
      output_type(node->get_output_element_type(0)) {}

void jit_convert_emitter::validate_types() const {
    OV_CPU_JIT_EMITTER_ASSERT(
        any_of(input_type, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
        "Unsupported input type: ",
        input_type.get_type_name());
    OV_CPU_JIT_EMITTER_ASSERT(
        any_of(output_type, ov::element::f32, ov::element::i32, ov::element::f16, ov::element::i8, ov::element::u8),
        "Unsupported output type: ",
        output_type.get_type_name());
}

size_t jit_convert_emitter::get_inputs_count() const {
    return 1;
}

void jit_convert_emitter::emit_data() const {
    jit_emitter::emit_data();
}

jit_convert_truncation_emitter::jit_convert_truncation_emitter(jit_generator_t* host,
                                                               cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node,
                                                               ov::element::Type exec_prc)
    : jit_convert_emitter(host, host_isa, node, exec_prc) {}

void jit_convert_truncation_emitter::emit_impl(const std::vector<size_t>& in_idxs,
                                               const std::vector<size_t>& out_idxs) const {
    validate_types();
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_convert_truncation_emitter::emit_isa(const std::vector<size_t>& in_idxs,
                                              const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    jit_conversion::emit_convert_process(h, src, dst, input_type, output_type, false);
}

jit_convert_saturation_emitter::jit_convert_saturation_emitter(jit_generator_t* host,
                                                               cpu_isa_t host_isa,
                                                               const std::shared_ptr<ov::Node>& node,
                                                               ov::element::Type exec_prc)
    : jit_convert_emitter(host, host_isa, node, exec_prc) {}

void jit_convert_saturation_emitter::emit_impl(const std::vector<size_t>& in_idxs,
                                               const std::vector<size_t>& out_idxs) const {
    validate_types();
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_idxs, out_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Unsupported ISA ", host_isa_);
    }
}

template <cpu_isa_t isa>
void jit_convert_saturation_emitter::emit_isa(const std::vector<size_t>& in_idxs,
                                              const std::vector<size_t>& out_idxs) const {
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    auto src = TReg(in_idxs[0]);
    auto dst = TReg(out_idxs[0]);
    jit_conversion::emit_convert_process(h, src, dst, input_type, output_type, true);
}

}  // namespace ov::intel_cpu::aarch64
