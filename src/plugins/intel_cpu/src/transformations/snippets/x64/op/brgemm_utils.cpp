// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_utils.hpp"

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "utils/general_utils.h"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

cpu_isa_t get_primitive_isa(const ov::element::Type& dt_in0, bool is_with_amx) {
    auto isa = isa_undef;
#define SUPPORT(X, Y) if (mayiuse(X)) { isa = X; } else { Y }
#define SUPPORT_ONE(X, MESSAGE) SUPPORT(X, OV_CPU_JIT_EMITTER_THROW(MESSAGE);)
#define SUPPORT_TWO(X, Y, MESSAGE) SUPPORT(X, SUPPORT_ONE(Y, MESSAGE))

    // Note: AMX might be not used even if it's supported by the hardware, check the BrgemmToBrgemmCPU pass for details
    if (is_with_amx) {
        SUPPORT_ONE(avx512_core_amx, "Unsupported hardware configuration: amx is supported only on avx512 platforms")
    } else if (dt_in0 == ov::element::bf16) {
        SUPPORT_ONE(avx512_core_bf16, "Unsupported hardware configuration: bf16 is supported only on avx512 platforms")
    } else if (one_of(dt_in0, ov::element::u8, ov::element::i8)) {
        SUPPORT_TWO(avx512_core_vnni, avx2_vnni, "Unsupported hardware configuration: int8 is supported only on vnni platforms")
    } else {
        SUPPORT_TWO(avx512_core, cpu::x64::avx2, "Unsupported hardware configuration: brgemm requires at least avx2 isa")
    }
    return isa;
#undef SUPPORT_TWO
#undef SUPPORT_ONE
#undef SUPPORT
}

BRGEMM_TYPE get_brgemm_type(const ov::element::Type& element_type_a, const Dimension& K_dim, const Dimension& N_dim, bool transpose_b) {
    if (element_type_a == element::f32)
        return transpose_b ? BRGEMM_TYPE::REPACKING_ONLY : BRGEMM_TYPE::STAND_ALONE;

    OPENVINO_ASSERT(element_type_a != element::bf16 || mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16),
                    "BF16 precision is not supported on this hardware");

    const auto brgemmVNNIFactor = 4 / element_type_a.size();
    if (one_of(element_type_a, element::u8, element::i8, element::bf16) &&
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) &&
        K_dim.is_static() && K_dim.get_length() % brgemmVNNIFactor == 0 &&
        N_dim.is_static() && N_dim.get_length() % brgemmVNNIFactor == 0)
        return BRGEMM_TYPE::WITH_AMX;
    // Note: this condition reproduces logic from the OneDNN Brgemm implementation. This is needed to align with the
    // backend requirements. More details in onednn/src/cpu/x64/brgemm/brgemm_utils.cpp
    if (element_type_a == ov::element::i8 &&
       !dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
       return BRGEMM_TYPE::WITH_COMPENSATIONS;

    if (one_of(element_type_a, element::u8, ov::element::bf16))
        return BRGEMM_TYPE::REPACKING_ONLY;
    OV_CPU_JIT_EMITTER_THROW("Failed to determine brgemm mode");
}

size_t compute_vnni_factor(const ov::element::Type& precision) {
    return data_type_vnni_granularity(static_cast<dnnl_data_type_t>(ov::intel_cpu::DnnlExtensionUtils::ElementTypeToDataType(precision)));
}

size_t get_elems_in_vec(const ov::element::Type& precision) {
    using namespace dnnl::impl::cpu;
    OV_CPU_JIT_EMITTER_ASSERT(x64::mayiuse(x64::avx2), "doesn't support non avx512 platforms");
    const auto vlen = x64::mayiuse(avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen : x64::cpu_isa_traits<x64::avx2>::vlen;
    return vlen / precision.size();
}

namespace repacking {
size_t get_repacking_buffer_size(const ov::snippets::lowered::ExpressionPtr& copy_b_expr) {
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::BrgemmCopyB>(copy_b_expr->get_node()));
    const auto& in_desc = copy_b_expr->get_input_port_descriptor(0);
    const auto& in_layout = in_desc->get_layout();
    const auto& in_subtensor = ov::snippets::utils::get_projected_subtensor(copy_b_expr->get_input_port(0));

    const size_t n_blk = *in_subtensor.rbegin();
    const size_t k_blk = *++in_subtensor.rbegin();
    OPENVINO_ASSERT(!is_dynamic_value(n_blk) && !is_dynamic_value(k_blk), "get_repacking_buffer_size must be called with static subtensor values");

    const auto& precision = copy_b_expr->get_node()->get_input_element_type(0);
    // Repacking buffer shape is set in accordance to OneDNN requirements
    const size_t N_dim = std::max(n_blk, compute_inner_n_block(precision));
    if (!in_layout.empty() && in_layout.back() != in_layout.size() - 1) {
        // In case of transpose, K dimension must be rounded-up to number of elems in vector register
        // For the details, please see 'transpose16x8' and 'fixup16x16' implementations and usage in onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
        const auto elems_in_vec = brgemm_utils::get_elems_in_vec(precision);
        return N_dim * rnd_up(k_blk, elems_in_vec);
    } else {
        // Low precision repacking writes the result by m_brgemmVNNIFactor * m_inner_n_block blocks
        // despite the actual size of the input data. Because of that we have to round-up the allocation shape to always have enough memory allocated.
        // For the details, please see 'copy_4x64' and 'copy_2x32' implementations and usage in onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
        const auto brgemmVNNIFactor = brgemm_utils::compute_vnni_factor(precision);
        OPENVINO_ASSERT(brgemmVNNIFactor > 0, "brgemmVNNIFactor value must be positive.");
        return N_dim * rnd_up(k_blk, brgemmVNNIFactor);
    }
}

size_t get_compensations_buffer_size(const ov::snippets::lowered::ExpressionPtr& copy_b_expr) {
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::BrgemmCopyB>(copy_b_expr->get_node()));
    const auto& in_subtensor = ov::snippets::utils::get_projected_subtensor(copy_b_expr->get_input_port(0));
    const size_t n_blk = *in_subtensor.rbegin();
    OPENVINO_ASSERT(!is_dynamic_value(n_blk), "get_compensations_buffer_size must be called with static subtensor values");
    const auto& precision = copy_b_expr->get_node()->get_input_element_type(0);
    // Compensations are computed during repacking, so we need to round-up allocation shape according to m_inner_n_block
    // because of OneDNN implementation nuances (as in get_repacking_buffer_size).
    // However, the compensations are computed by N dimension, so K dimension doesn't affect the compensations buffer
    return std::max(n_blk, compute_inner_n_block(precision));
}

size_t compute_out_leading_dim(const size_t n_block, const ov::element::Type& precision) {
    return std::max(n_block, compute_inner_n_block(precision));
}

size_t compute_inner_n_block(const ov::element::Type& precision) {
    switch (precision) {
        case element::i8: return 64;
        case element::bf16: return 32;
        case element::f32: return 16;
        default: OPENVINO_THROW("BrgemmCopyB doesn't support precision ", precision);
    }
}
}   // namespace repacking
}   // namespace brgemm_utils
}   // namespace intel_cpu
template <>
EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>& EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>::get() {
    static auto enum_names =
            EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>("ov::intel_cpu::jit_bgremm_utils::BRGEMM_TYPE",
                                                                {{"stand_alone", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::STAND_ALONE},
                                                                 {"with_amx", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::WITH_AMX},
                                                                 {"with_compensations", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::WITH_COMPENSATIONS},
                                                                 {"repacking_only", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::REPACKING_ONLY}});
    return enum_names;
}
}   // namespace ov
