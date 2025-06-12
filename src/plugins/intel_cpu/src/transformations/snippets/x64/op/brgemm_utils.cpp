// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_utils.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <string>

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_cpu_blocking.hpp"
#include "transformations/snippets/x64/pass/lowered/expressions/brgemm_copy_b_buffer_expressions.hpp"
#include "utils/general_utils.h"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::snippets::utils;

namespace ov {

namespace intel_cpu::brgemm_utils {

BrgemmConfig::BrgemmConfig(const ov::element::Type& src_dt,
                           const ov::element::Type& wei_dt,
                           bool are_wei_constant,
                           bool transposed_b)
    : BrgemmConfig(get_prim_isa(src_dt, wei_dt), src_dt, wei_dt, are_wei_constant, transposed_b) {}

// [TODO] 168764: Blocked weights repacking requires blocked loop by N for correct ptr increments.
//        If kn_blocking is not supported by Brgemm, we cannot repack weights to blocked layout
BrgemmConfig::BrgemmConfig(const dnnl::impl::cpu::x64::cpu_isa_t& isa,
                           const ov::element::Type& src_dt,
                           const ov::element::Type& wei_dt,
                           bool are_wei_constant,
                           bool transposed_b)
    : m_isa(isa),
      m_with_compensations(src_dt == ov::element::i8 && !one_of(m_isa, avx512_core_amx, avx2_vnni_2)),
      m_are_wei_constant(are_wei_constant),
      m_are_wei_blocked(ov::intel_cpu::pass::BrgemmCPUBlocking::is_kn_blocking_supported(src_dt) && m_are_wei_constant),
      m_wei_k_blk(get_elems_in_vec(wei_dt)) {
    const auto is_fp32 = src_dt == ov::element::f32 && wei_dt == ov::element::f32;

    // FC always requires weight repacking
    m_with_wei_repacking = !is_fp32 || transposed_b || m_are_wei_constant || m_are_wei_blocked;

    // TODO: Add more logic based on shapes and prc
    if (m_are_wei_blocked) {
        m_wei_n_blk = is_superset(m_isa, avx512_core) ? 64 : 48;
    } else {
        switch (wei_dt) {
        case element::i8:
            m_wei_n_blk = 64;
            break;
        case element::bf16:
        case element::f16:
            m_wei_n_blk = 32;
            break;
        case element::f32:
            m_wei_n_blk = 16;
            break;
        default:
            OPENVINO_THROW("Unsupport precision of weights", wei_dt);
        }
    }

    validate();
}

dnnl::impl::cpu::x64::cpu_isa_t BrgemmConfig::get_prim_isa(const ov::element::Type& src_dt,
                                                           const ov::element::Type& wei_dt) {
#define RETURN_IF_SUPPORTED(x) \
    if (mayiuse(x)) {          \
        return x;              \
    }

    const auto is_fp32 = src_dt == ov::element::f32 && wei_dt == ov::element::f32;
    const auto is_fp16 = src_dt == ov::element::f16 && wei_dt == ov::element::f16;
    const auto is_bf16 = src_dt == ov::element::bf16 && wei_dt == ov::element::bf16;
    const auto is_int8 =
        ov::snippets::utils::one_of(src_dt, ov::element::i8, ov::element::u8) && wei_dt == ov::element::i8;
    OPENVINO_ASSERT(is_fp32 || is_fp16 || is_bf16 || is_int8,
                    "Incorrect configuration: src_dt = ",
                    src_dt,
                    ", wei_dt = ",
                    wei_dt);

    if (is_bf16) {
        RETURN_IF_SUPPORTED(avx512_core_amx)
        RETURN_IF_SUPPORTED(avx512_core_bf16)
        RETURN_IF_SUPPORTED(avx2_vnni_2)
        return isa_undef;
    }

    if (is_fp16) {
        RETURN_IF_SUPPORTED(avx512_core_amx_fp16)
        RETURN_IF_SUPPORTED(avx2_vnni_2)
        return isa_undef;
    }

    if (is_int8) {
        RETURN_IF_SUPPORTED(avx512_core_amx)
        RETURN_IF_SUPPORTED(avx512_core_vnni)
        RETURN_IF_SUPPORTED(avx2_vnni_2)
        RETURN_IF_SUPPORTED(avx2_vnni)
        return isa_undef;
    }

    RETURN_IF_SUPPORTED(avx512_core)
    RETURN_IF_SUPPORTED(cpu::x64::avx2)
    return isa_undef;
#undef RETURN_IF_SUPPORTED
}

bool BrgemmConfig::is_amx() const {
    return is_superset(m_isa, cpu_isa_t::amx_tile);
}

void BrgemmConfig::validate() const {
    OPENVINO_ASSERT(m_isa != isa_undef, "ISA is undefined");
    OPENVINO_ASSERT(ov::snippets::utils::implication(m_with_compensations, !is_amx() && m_with_wei_repacking),
                    "Compensations must be only with BrgemmCopyB on non-amx platforms");
    OPENVINO_ASSERT(m_wei_n_blk > 0 && m_wei_k_blk > 0, "Weight block sizes must be positive");
}

size_t BrgemmConfig::get_elems_in_vec(const ov::element::Type& precision) {
    using namespace dnnl::impl::cpu;
    OV_CPU_JIT_EMITTER_ASSERT(x64::mayiuse(x64::avx2), "doesn't support non avx512 platforms");
    const auto vlen =
        x64::mayiuse(avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen : x64::cpu_isa_traits<x64::avx2>::vlen;
    return vlen / precision.size();
}

size_t compute_vnni_factor(const ov::element::Type& precision) {
    return data_type_vnni_granularity(
        static_cast<dnnl_data_type_t>(ov::intel_cpu::DnnlExtensionUtils::ElementTypeToDataType(precision)));
}

namespace repacking {
ov::snippets::VectorDims compute_buffer_b_allocation_shape(size_t K, size_t N, size_t wei_k_blk, size_t wei_n_blk) {
    OPENVINO_ASSERT(
        !ov::snippets::utils::is_dynamic_value(wei_k_blk) && !ov::snippets::utils::is_dynamic_value(wei_n_blk),
        "wei_k_blk and wei_n_blk cannot be dynamic");

    const size_t new_N = compute_blocked_dim(N, wei_n_blk);
    return ov::snippets::VectorDims{ov::snippets::utils::div_up(K, wei_k_blk), new_N, wei_k_blk};
}

ov::snippets::lowered::ExpressionPtr get_copy_b_expr(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) {
    OPENVINO_ASSERT(ov::is_type<BrgemmCPU>(brgemm_expr->get_node()),
                    "get_copy_b_expr must be called only for BrgemmCPU node");
    auto b_input_expr = brgemm_expr->get_input_port_connector(1)->get_source().get_expr();
    if (ov::is_type<BrgemmCopyB>(b_input_expr->get_node())) {
        return b_input_expr;
    }
    if (ov::is_type<RepackedWeightsBufferExpression>(b_input_expr)) {
        OPENVINO_ASSERT(b_input_expr->get_input_count() >= 1,
                        "RepackedWeightsBufferExpression on brgemm's B input must have at least one input");
        auto input_buffer_expr = b_input_expr->get_input_port_connector(0)->get_source().get_expr();
        if (ov::is_type<BrgemmCopyB>(input_buffer_expr->get_node())) {
            return input_buffer_expr;
        }
    }
    return nullptr;
}
}  // namespace repacking
}  // namespace intel_cpu::brgemm_utils

bool AttributeAdapter<ov::intel_cpu::brgemm_utils::BrgemmConfig>::visit_attributes(AttributeVisitor& visitor) {
    bool with_wei_repacking = m_ref.with_wei_repacking();
    bool with_comps = m_ref.with_compensations();
    bool are_wei_blocked = m_ref.are_wei_blocked();
    bool are_wei_constant = m_ref.are_wei_constant();
    bool is_amx = m_ref.is_amx();
    std::string isa = JIT_IMPL_NAME_HELPER("", m_ref.isa(), "");
    size_t wei_n_blk = m_ref.wei_n_blk();
    size_t wei_k_blk = m_ref.wei_k_blk();

    visitor.on_attribute("with_brgemm_copy_b", with_wei_repacking);
    visitor.on_attribute("with_compensations", with_comps);
    visitor.on_attribute("are_wei_blocked", are_wei_blocked);
    visitor.on_attribute("are_wei_constant", are_wei_constant);
    visitor.on_attribute("is_amx", is_amx);
    visitor.on_attribute("prim_isa", isa);
    visitor.on_attribute("wei_n_blk", wei_n_blk);
    visitor.on_attribute("wei_k_blk", wei_k_blk);
    return true;
}
}  // namespace ov
