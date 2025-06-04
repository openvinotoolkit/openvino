// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_types.h"
#include "emitters/tpp/common/kernel_executors/brgemm.hpp"
#include "emitters/tpp/x64/jit_tpp_emitter.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

using jit_generator_t = dnnl::impl::cpu::x64::jit_generator_t;
using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

namespace ov::intel_cpu {

void BrgemmTppEmitter::validate_subtensors(const VectorDims& in_0, const VectorDims& in_1, const VectorDims& out_0) {
    bool subtensors_compatible = in_0.size() == in_1.size() && in_0.size() == out_0.size() && in_0.size() == 2 &&
                                 in_0[1] == in_1[0] && in_0[0] == out_0[0] && in_1[1] == out_0[1];
    OV_CPU_JIT_EMITTER_ASSERT(subtensors_compatible, "Incompatible subtensors");
}

BrgemmTppEmitter::BrgemmTppEmitter(jit_generator_t* h,
                                   cpu_isa_t isa,
                                   const ExpressionPtr& expr,
                                   const snippets::KernelExecutorTablePtr& kernel_table,
                                   const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : TppEmitter(h, isa, expr) {
    const auto& brgemm_node = as_type_ptr<intel_cpu::tpp::op::BrgemmTPP>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node && !brgemm_node->is_dynamic(), "Invoked with invalid node type");
    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    tpp::BrgemmKernelConfig kernel_config(brg0Prc, brg1Prc);
    m_kernel_executor =
        kernel_table->register_kernel<tpp::BrgemmKernelExecutor>(expr, compiled_kernel_cache, kernel_config);
}

std::set<std::vector<element::Type>> BrgemmTppEmitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    // Note: BrgemmTpp currently supports only fp32
    return {{element::f32, element::f32}};
}

void BrgemmTppEmitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input regs, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
}

const uintptr_t BrgemmTppEmitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor.get());
}

const uintptr_t BrgemmTppEmitter::get_execute_function_ptr() const {
    return reinterpret_cast<const uintptr_t>(tpp::BrgemmKernelExecutor::execute);
}

}  // namespace ov::intel_cpu
