// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

#include "openvino/core/type/element_type.hpp"

namespace ov::element {

struct TypeInfo {
    size_t m_bitwidth;
    bool m_is_real;
    bool m_is_signed;
    bool m_is_quantized;
    const char* m_cname;
    const char* m_type_name;
    const char* const* m_aliases;
    size_t m_alias_count;

    bool has_name(const std::string& type) const {
        if (type == m_type_name) {
            return true;
        } else {
            const auto last = m_aliases + m_alias_count;
            return std::find(m_aliases, last, type) != last;
        }
    }

    constexpr bool is_valid() const {
        return m_cname != nullptr && m_type_name != nullptr;
    }
};

/**
 * @brief Get TypeInfo of given element type.
 *
 * @param type Openvino element type to get its description.
 * @return Reference to TypeInfo.
 */
OPENVINO_API const TypeInfo& get_type_info(Type_t type);

}  // namespace ov::element
