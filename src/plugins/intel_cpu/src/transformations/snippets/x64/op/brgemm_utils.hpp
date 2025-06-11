// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {

namespace intel_cpu::brgemm_utils {

class BrgemmConfig {
public:
    BrgemmConfig() = default;
    BrgemmConfig(const ov::element::Type& src_dt,
                 const ov::element::Type& wei_dt,
                 bool are_wei_constant,
                 bool transposed_b);
    BrgemmConfig(const dnnl::impl::cpu::x64::cpu_isa_t& isa,
                 const ov::element::Type& src_dt,
                 const ov::element::Type& wei_dt,
                 bool are_wei_constant,
                 bool transposed_b);

    dnnl::impl::cpu::x64::cpu_isa_t isa() const {
        return m_isa;
    }
    bool is_amx() const;
    bool with_wei_repacking() const {
        return m_with_wei_repacking;
    }
    bool with_compensations() const {
        return m_with_compensations;
    }
    bool with_scratchpad() const {
        return is_amx() || with_compensations();
    }
    bool are_wei_blocked() const {
        return m_are_wei_blocked;
    }
    bool are_wei_constant() const {
        return m_are_wei_constant;
    }

    size_t wei_n_blk() const {
        return m_wei_n_blk;
    }
    size_t wei_k_blk() const {
        return m_wei_k_blk;
    }

private:
    void validate() const;

    static dnnl::impl::cpu::x64::cpu_isa_t get_prim_isa(const ov::element::Type& src_dt,
                                                        const ov::element::Type& wei_dt);
    static size_t get_elems_in_vec(const ov::element::Type& precision);

    dnnl::impl::cpu::x64::cpu_isa_t m_isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
    bool m_with_wei_repacking = false;
    bool m_with_compensations = false;
    bool m_are_wei_constant = false;
    /* Currently we support the following representations of weights:
     *  - planar - ab
     *  - planar with inner K blocking for low precision - Ab<vnni_factor>a
     *  - blocked - BA<wei_k_blk>a<m_wei_n_blk>b<vnni_factor>a (if there is vnni)
     *  "Blocked" weight-format helps to achieve better cache utilization - LDB is equal to <m_wei_n_blk>.
     * Note: FC requires blocked by N weights for better cache utilization (small LDB).
     *       In MatMul scenario it might lead to perf degradation.
     */
    bool m_are_wei_blocked = false;
    size_t m_wei_n_blk = 0lu;
    size_t m_wei_k_blk = 0lu;
};

/// \brief Computes VNNI factor used by OneDNN implementation. Depends on tensor precision
size_t compute_vnni_factor(const ov::element::Type& precision);

/// \brief The following helpers return True if the target precision is supported by BRGEMM on the current platform
inline bool is_fp32_supported() {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2);
}
inline bool is_bf16_supported() {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) ||
           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16) ||
           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2);
}
inline bool is_fp16_supported() {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx_fp16) ||
           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2);
}
inline bool is_i8_supported() {
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx) ||
           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_vnni) ||
           dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni);
}

namespace repacking {
/// \brief  Computes Blocked (N/K) dim in output blocked shape of BrgemmCopyB
template <typename T,
          typename = typename std::enable_if_t<(std::is_same_v<T, size_t> || std::is_same_v<T, int64_t>), bool>>
inline T compute_blocked_dim(T dim, size_t blk) {
    assert(!ov::snippets::utils::is_dynamic_value(blk) && "blk cannot be dynamic");
    return ov::snippets::utils::rnd_up(dim, static_cast<T>(blk));
}

/// \brief  Computes LDB
template <typename T,
          typename = typename std::enable_if_t<(std::is_same_v<T, size_t> || std::is_same_v<T, int64_t>), bool>>
inline T compute_LDB(T n, size_t wei_n_blk, bool are_wei_blocked) {
    assert(!ov::snippets::utils::is_dynamic_value(wei_n_blk) && "wei_n_blk cannot be dynamic");
    return are_wei_blocked ? wei_n_blk : compute_blocked_dim(n, wei_n_blk);
}

/// \brief  Computes allocation shape for Buffer between BrgemmCopyB and Brgemm
ov::snippets::VectorDims compute_buffer_b_allocation_shape(size_t K, size_t N, size_t wei_k_blk, size_t wei_n_blk);

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
class AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig> : public VisitorAdapter {
public:
    AttributeAdapter(intel_cpu::brgemm_utils::BrgemmConfig& ref) : m_ref(ref) {}
    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig>");

protected:
    intel_cpu::brgemm_utils::BrgemmConfig& m_ref;
};
}  // namespace ov
