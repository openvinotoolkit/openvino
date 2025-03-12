// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {

namespace intel_cpu::brgemm_utils {

enum class BRGEMM_TYPE {
    STAND_ALONE,  // No extra requirements, used for f32|f32
    WITH_AMX,     // i8|i8 or bf16|bf16 on AMX system or fp16|fp16 on AMX_FP16 system - needs BrgemmCopyB and scratchpad
    WITH_COMPENSATIONS,  // i8|i8 (non-AMX system) - needs BrgemmCopyB for data repacking and compensations
    REPACKING_ONLY,      // u8|i8, or bf16|bf16 (non-AMX system), or brgemm with transpose_b=true - needs BrgemmCopyB on
                         // second input for data repacking
};

dnnl::impl::cpu::x64::cpu_isa_t get_primitive_isa(const ov::element::Type& dt_in0, bool is_with_amx);

BRGEMM_TYPE get_brgemm_type(const element::Type& element_type_a, bool transpose_b);

inline bool stand_alone(BRGEMM_TYPE type) {
    return type == BRGEMM_TYPE::STAND_ALONE;
}

inline bool with_amx(BRGEMM_TYPE type) {
    return type == BRGEMM_TYPE::WITH_AMX;
}

inline bool with_compensations(BRGEMM_TYPE type) {
    return type == BRGEMM_TYPE::WITH_COMPENSATIONS;
}

inline bool repacking_only(BRGEMM_TYPE type) {
    return type == BRGEMM_TYPE::REPACKING_ONLY;
}

inline bool with_repacking(BRGEMM_TYPE type) {
    return type != BRGEMM_TYPE::STAND_ALONE;
}

inline bool with_scratchpad(BRGEMM_TYPE type) {
    return with_compensations(type) || with_amx(type);
}

/// \brief Computes VNNI factor used by OneDNN implementation. Depends on tensor precision
size_t compute_vnni_factor(const ov::element::Type& precision);
/// \brief Computes number of elems with requested precision that fit in the vector register
size_t get_elems_in_vec(const ov::element::Type& precision);

namespace repacking {
/// \brief  Computes inner N block size used by OneDNN implementation. Depends on tensor precision
size_t compute_inner_n_block(const ov::element::Type& precision);
/// \brief  Computes inner K block size used by OneDNN implementation. Depends on tensor precision
size_t compute_inner_k_block(const ov::element::Type& precision);

/// \brief  Computes N dim in output blocked shape of BrgemmCopyB. Depends on tensor precision
template <typename T,
          typename = typename std::enable_if_t<(std::is_same_v<T, size_t> || std::is_same_v<T, int64_t>), bool>>
inline T compute_repacked_n_dim(T n, const ov::element::Type& precision) {
    return ov::snippets::utils::rnd_up(n, static_cast<T>(compute_inner_n_block(precision)));
}

/// \brief  Computes allocation shape for Buffer between BrgemmCopyB and Brgemm
ov::snippets::VectorDims compute_buffer_b_allocation_shape(size_t K,
                                                           size_t N,
                                                           const ov::element::Type& prc,
                                                           bool is_transposed);

/**
 * @brief Retrieves the expression pointer for the brgemm_copy_b expression corresponding to the given BrgemmCPU
 * expression.
 * @param brgemm_expr The expression pointer for the BrgemmCPU operation.
 * @return The expression pointer for the BrgemmCopyB operation.
 */
snippets::lowered::ExpressionPtr get_copy_b_expr(const snippets::lowered::ExpressionPtr& brgemm_expr);
}  // namespace repacking
}  // namespace intel_cpu::brgemm_utils

template <>
class AttributeAdapter<intel_cpu::brgemm_utils::BRGEMM_TYPE>
    : public EnumAttributeAdapterBase<intel_cpu::brgemm_utils::BRGEMM_TYPE> {
public:
    AttributeAdapter(intel_cpu::brgemm_utils::BRGEMM_TYPE& value)
        : EnumAttributeAdapterBase<intel_cpu::brgemm_utils::BRGEMM_TYPE>(value) {}
    OPENVINO_RTTI("AttributeAdapter<ov::intel_cpu::jit_brgemm_utils::BRGEMM_TYPE>");
};
}  // namespace ov
