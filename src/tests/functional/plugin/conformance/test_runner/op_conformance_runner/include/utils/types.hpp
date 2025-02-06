
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace op_conformance {

static std::set<std::string> get_element_type_names() {
    std::vector<ov::element::Type> element_types = {ov::element::Type_t::f64,
                                                    ov::element::Type_t::f32,
                                                    ov::element::Type_t::f16,
                                                    ov::element::Type_t::bf16,
                                                    ov::element::Type_t::nf4,
                                                    ov::element::Type_t::i64,
                                                    ov::element::Type_t::i32,
                                                    ov::element::Type_t::i16,
                                                    ov::element::Type_t::i8,
                                                    ov::element::Type_t::i4,
                                                    ov::element::Type_t::u64,
                                                    ov::element::Type_t::u32,
                                                    ov::element::Type_t::u16,
                                                    ov::element::Type_t::u8,
                                                    ov::element::Type_t::u4,
                                                    ov::element::Type_t::u1,
                                                    ov::element::Type_t::boolean,
                                                    ov::element::Type_t::dynamic};
    OPENVINO_SUPPRESS_DEPRECATED_START
    element_types.emplace_back(element::undefined);
    OPENVINO_SUPPRESS_DEPRECATED_END
    std::set<std::string> result;
    for (const auto& element_type : element_types) {
        std::string element_name = element_type.get_type_name();
        std::transform(element_name.begin(), element_name.end(), element_name.begin(),
                       [](unsigned char symbol){ return std::tolower(symbol); });
        result.insert(element_name);
    }
    return result;
}

static auto element_type_names = get_element_type_names();


} // namespace op_conformance
} // namespace test
} // namespace ov
