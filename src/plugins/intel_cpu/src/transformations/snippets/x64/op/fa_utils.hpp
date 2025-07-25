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
#include "snippets/op/subgraph.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {

namespace intel_cpu::fa_utils {

class FAConfig {
public:
    FAConfig() = default;
    FAConfig(const ov::element::Type& src_dt,
                 const ov::element::Type& wei_dt,
                 const ov::element::Type& orig_wei_dt,
                 bool transposed_b);
    FAConfig(const dnnl::impl::cpu::x64::cpu_isa_t& isa,
                 const ov::element::Type& src_dt,
                 const ov::element::Type& wei_dt,
                 const ov::element::Type& orig_wei_dt,
                 bool transposed_b);

    [[nodiscard]] dnnl::impl::cpu::x64::cpu_isa_t isa() const {
        return m_isa;
    }
    [[nodiscard]] ov::element::Type src_dt() const {
        return m_src_dt;
    }
    [[nodiscard]] ov::element::Type wei_dt() const {
        return m_wei_dt;
    }
    [[nodiscard]] ov::element::Type orig_wei_dt() const {
        return m_orig_wei_dt;
    }
    [[nodiscard]] bool transposed_b() const {
        return m_transposed_b;
    }
    [[nodiscard]] size_t q_len_blk() const {
        return m_q_len_blk;
    }
    [[nodiscard]] size_t kv_len_blk() const {
        return m_kv_len_blk;
    }

private:
    void validate() const;

    static dnnl::impl::cpu::x64::cpu_isa_t get_prim_isa(const ov::element::Type& src_dt,
                                                        const ov::element::Type& wei_dt);

    dnnl::impl::cpu::x64::cpu_isa_t m_isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;

    ov::element::Type m_src_dt = ov::element::dynamic;
    ov::element::Type m_wei_dt = ov::element::dynamic;
    ov::element::Type m_orig_wei_dt = ov::element::dynamic;

    bool m_transposed_b = false;
    size_t m_q_len_blk = 0lu;
    size_t m_kv_len_blk = 0lu;
};

}  // namespace intel_cpu::fa_utils

}  // namespace ov
