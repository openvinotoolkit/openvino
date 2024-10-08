// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov {
namespace intel_cpu {
namespace brgemm_utils {

class BrgemmConfig {
public:
    BrgemmConfig() = default;
    BrgemmConfig(const ov::element::Type& src_dt, const ov::element::Type& wei_dt, size_t K, bool transposed_b);
    BrgemmConfig(const ov::element::Type& src_dt, dnnl::impl::cpu::x64::cpu_isa_t isa,
                 bool need_copy_a = false, bool need_copy_b = false, bool need_compensations = false, bool need_wsp = false);

    dnnl::impl::cpu::x64::cpu_isa_t isa() const { return m_isa; }
    bool is_amx() const { return m_isa == dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_amx; }
    bool need_copy_a() const { return m_need_copy_a; }
    bool need_copy_b() const { return m_need_copy_b; }
    bool need_compensations() const { return m_need_compensations; }
    bool need_wsp() const { return m_need_wsp; }

private:
    void validate() const;

    dnnl::impl::cpu::x64::cpu_isa_t m_isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
    bool m_need_copy_a = false;
    bool m_need_copy_b = false;
    bool m_need_compensations = false;
    bool m_need_wsp = false;
};

/// \brief Computes VNNI factor used by OneDNN implementation. Depends on tensor precision
size_t compute_vnni_factor(const ov::element::Type& precision);
/// \brief Computes number of elems with requested precision that fit in the vector register
size_t get_elems_in_vec(const ov::element::Type& precision);

namespace repacking {
/**
 * @brief Computes leading dimension (LDA) which must be used in brgemm and brgemm_copy_a emitters
 * @param k_block K block size shared between BrgemmCPU and BrgemmCopyA node
 * @param precision tensor precision
 */
size_t compute_LDA(const size_t k_block, const ov::element::Type& precision);
/**
 * @brief Computes leading dimension (LDB) which must be used in brgemm and brgemm_copy_b emitters
 * @param n_block N block size shared between BrgemmCPU and BrgemmCopyB node
 * @param precision tensor precision
 */
size_t compute_LDB(const size_t n_block, const ov::element::Type& precision);
/// \brief  Computes inner N block size used by OneDNN implementation. Depends on tensor precision
size_t compute_inner_n_block(const ov::element::Type& precision);
/// \brief  Computes inner K block size used by OneDNN implementation. Depends on tensor precision
size_t compute_inner_k_block(const ov::element::Type& precision);
}   // namespace repacking
}   // namespace brgemm_utils
}   // namespace intel_cpu

template <>
class AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig> : public VisitorAdapter {
public:
    AttributeAdapter(intel_cpu::brgemm_utils::BrgemmConfig& ref) : m_ref(ref) {}
    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<intel_cpu::brgemm_utils::BrgemmConfig>");

protected:
    intel_cpu::brgemm_utils::BrgemmConfig& m_ref;
};
}   // namespace ov
