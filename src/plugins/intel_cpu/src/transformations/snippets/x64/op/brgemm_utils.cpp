// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_utils.hpp"

#include "dnnl_extension_utils.h"
#include "emitters/utils.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/buffer.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "utils/general_utils.h"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::snippets::utils;

namespace ov {

namespace intel_cpu::brgemm_utils {

cpu_isa_t get_primitive_isa(const ov::element::Type& dt_in0, bool is_with_amx) {
    auto isa = isa_undef;
#define SUPPORT(X, Y) \
    if (mayiuse(X)) { \
        isa = X;      \
    } else {          \
        Y             \
    }
#define SUPPORT_ONE(X, MESSAGE)         SUPPORT(X, OV_CPU_JIT_EMITTER_THROW(MESSAGE);)
#define SUPPORT_TWO(X, Y, MESSAGE)      SUPPORT(X, SUPPORT_ONE(Y, MESSAGE))
#define SUPPORT_THREE(X, Y, Z, MESSAGE) SUPPORT(X, SUPPORT_TWO(Y, Z, MESSAGE))

    // Note: AMX might be not used even if it's supported by the hardware, check the BrgemmToBrgemmCPU pass for details
    if (is_with_amx) {
        if (dt_in0 == ov::element::f16) {
            SUPPORT_ONE(avx512_core_amx_fp16,
                        "Unsupported hardware configuration: amx is supported only on avx512 platforms")
        } else
            SUPPORT_ONE(avx512_core_amx,
                        "Unsupported hardware configuration: amx is supported only on avx512 platforms")
    } else if (dt_in0 == ov::element::bf16) {
        SUPPORT_ONE(avx512_core_bf16, "Unsupported hardware configuration: bf16 is supported only on avx512 platforms")
    } else if (one_of(dt_in0, ov::element::u8, ov::element::i8)) {
        SUPPORT_THREE(avx512_core_vnni,
                      avx2_vnni_2,
                      avx2_vnni,
                      "Unsupported hardware configuration: int8 is supported only on vnni platforms")
    } else {
        SUPPORT_TWO(avx512_core,
                    cpu::x64::avx2,
                    "Unsupported hardware configuration: brgemm requires at least avx2 isa")
    }
    return isa;
#undef SUPPORT_TWO
#undef SUPPORT_ONE
#undef SUPPORT
}

BRGEMM_TYPE get_brgemm_type(const ov::element::Type& element_type_a, bool transpose_b) {
    if (element_type_a == element::f32) {
        return transpose_b ? BRGEMM_TYPE::REPACKING_ONLY : BRGEMM_TYPE::STAND_ALONE;
    }

    OPENVINO_ASSERT(element_type_a != element::bf16 || mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16),
                    "BrgemmCPU BF16 precision is not supported on non avx512_core_bf16 system");
    OPENVINO_ASSERT(element_type_a != element::f16 || mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16),
                    "BrgemmCPU FP16 precision is not supported on non avx512_core_amx_fp16 system");

    if (one_of(element_type_a, element::u8, element::i8, element::bf16) &&
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx)) {
        return BRGEMM_TYPE::WITH_AMX;
    }
    if (element_type_a == ov::element::f16 &&
        dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16)) {
        return BRGEMM_TYPE::WITH_AMX;
    }
    // Note: this condition reproduces logic from the OneDNN Brgemm implementation. This is needed to align with the
    // backend requirements. More details in onednn/src/cpu/x64/brgemm/brgemm_utils.cpp
    if (element_type_a == ov::element::i8) {
        return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2) ? BRGEMM_TYPE::REPACKING_ONLY
                                                                                : BRGEMM_TYPE::WITH_COMPENSATIONS;
    }

    if (one_of(element_type_a, element::u8, ov::element::bf16)) {
        return BRGEMM_TYPE::REPACKING_ONLY;
    }
    OV_CPU_JIT_EMITTER_THROW("Failed to determine brgemm mode");
}

size_t compute_vnni_factor(const ov::element::Type& precision) {
    return data_type_vnni_granularity(
        static_cast<dnnl_data_type_t>(ov::intel_cpu::DnnlExtensionUtils::ElementTypeToDataType(precision)));
}

size_t get_elems_in_vec(const ov::element::Type& precision) {
    using namespace dnnl::impl::cpu;
    OV_CPU_JIT_EMITTER_ASSERT(x64::mayiuse(x64::avx2), "doesn't support non avx512 platforms");
    const auto vlen =
        x64::mayiuse(avx512_core) ? x64::cpu_isa_traits<x64::avx512_core>::vlen : x64::cpu_isa_traits<x64::avx2>::vlen;
    return vlen / precision.size();
}

namespace repacking {
size_t compute_inner_n_block(const ov::element::Type& precision) {
    switch (precision) {
    case element::i8:
        return 64;
    case element::bf16:
    case element::f16:
        return 32;
    case element::f32:
        return 16;
    default:
        OPENVINO_THROW("BrgemmCopyB doesn't support precision ", precision);
    }
}

size_t compute_inner_k_block(const ov::element::Type& precision) {
    return brgemm_utils::get_elems_in_vec(precision);
}

ov::snippets::VectorDims compute_buffer_b_allocation_shape(size_t K,
                                                           size_t N,
                                                           const ov::element::Type& prc,
                                                           bool is_transposed) {
    const size_t new_N = compute_repacked_n_dim(N, prc);
    //  - In case of transpose, K dimension must be rounded-up to number of elems in vector register
    //    For the details, please see 'transpose16x8' and 'fixup16x16' implementations and usage in
    //    onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
    //  - Low precision repacking writes the result by VNNIFactor * wei_n_blk blocks
    //    despite the actual size of the input data. Because of that we have to round-up the allocation shape to always
    //    have enough memory allocated. For the details, please see 'copy_4x64' and 'copy_2x32' implementations and
    //    usage in onednn/src/cpu/x64/matmul/brgemm_matmul_copy_utils.cpp
    const size_t K_alignment =
        is_transposed ? brgemm_utils::get_elems_in_vec(prc) : brgemm_utils::compute_vnni_factor(prc);
    return ov::snippets::VectorDims{new_N, ov::snippets::utils::div_up(K, K_alignment), K_alignment};
}

ov::snippets::lowered::ExpressionPtr get_copy_b_expr(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) {
    OPENVINO_ASSERT(ov::is_type<BrgemmCPU>(brgemm_expr->get_node()),
                    "get_copy_b_expr must be called only for BrgemmCPU node");
    auto b_input_expr = brgemm_expr->get_input_port_connector(1)->get_source().get_expr();
    if (ov::is_type<BrgemmCopyB>(b_input_expr->get_node())) {
        return b_input_expr;
    }
    if (ov::is_type<snippets::lowered::BufferExpression>(b_input_expr)) {
        OPENVINO_ASSERT(b_input_expr->get_input_count() >= 1,
                        "BufferExpression on brgemm's B input must have at least one input");
        auto input_buffer_expr = b_input_expr->get_input_port_connector(0)->get_source().get_expr();
        if (ov::is_type<BrgemmCopyB>(input_buffer_expr->get_node())) {
            return input_buffer_expr;
        }
    }
    return nullptr;
}
}  // namespace repacking
}  // namespace intel_cpu::brgemm_utils

template <>
EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>& EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>::get() {
    static auto enum_names = EnumNames<ov::intel_cpu::brgemm_utils::BRGEMM_TYPE>(
        "ov::intel_cpu::jit_bgremm_utils::BRGEMM_TYPE",
        {{"stand_alone", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::STAND_ALONE},
         {"with_amx", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::WITH_AMX},
         {"with_compensations", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::WITH_COMPENSATIONS},
         {"repacking_only", ov::intel_cpu::brgemm_utils::BRGEMM_TYPE::REPACKING_ONLY}});
    return enum_names;
}
}  // namespace ov
