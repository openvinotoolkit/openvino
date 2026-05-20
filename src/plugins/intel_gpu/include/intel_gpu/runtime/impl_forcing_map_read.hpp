// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "openvino/core/any.hpp"

namespace ov::util {

// specific parser for ImplForcingMap supporting quotation marks interpretation. 
// it allows the keys to contain colons (:) which should not be treated as separators
// to be removed when the support for quotation marks is implemented in generic ov::Any
template <>
struct Read<ov::intel_gpu::ImplForcingMap> {
    void operator()(std::istream& is, ov::intel_gpu::ImplForcingMap& map) const {
        char c;

        is >> c;
        OPENVINO_ASSERT(c == '{',
                        "Failed to parse ov::intel_gpu::ImplForcingMap. Starting symbol is not '{', it's ", c);

        while (c != '}') {
            std::string key, value;
            if (is.peek() == '\'' || is.peek() == '"') {
                is >> c;
                const char separator = c;
                std::getline(is, key, separator);
                OPENVINO_ASSERT(is.get() == ':',
                                "Parsing error: Separator (:) needed after key name. format: {" +
                                    std::string(1, separator) + "key" + std::string(1, separator) + ":value}");
            } else {
                std::getline(is, key, ':');
            }

            size_t enclosed_container_level = 0;

            while (is.good()) {
                is >> c;
                if (c == ',') {
                    if (enclosed_container_level == 0)
                        break;
                }
                if (c == '{' || c == '[')
                    ++enclosed_container_level;
                if (c == '}' || c == ']') {
                    if (enclosed_container_level == 0)
                        break;
                    --enclosed_container_level;
                }

                value += c;
            }
            map.emplace(from_string<cldnn::primitive_id>(key), from_string<ov::intel_gpu::ImplementationDesc>(value));
        }

        OPENVINO_ASSERT(c == '}',
                        "Failed to parse ov::intel_gpu::ImplForcingMap. Ending symbol is not '}', it's ", c);
    }
};

}  // namespace ov::util