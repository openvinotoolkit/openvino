//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace intel_npu::vm {

inline void printIndent(size_t indentLevel) {
    constexpr auto numSpaces = 2;
    for (size_t i = 0; i < indentLevel * numSpaces; ++i) {
        std::cout << " ";
    }
}

std::string toHex(uint8_t byte);

template <typename T>
std::string formatVector(const std::vector<T>& values) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }
        stream << values.at(i);
    }
    stream << "]";
    return stream.str();
}

std::string formatPtrVectorContent(const std::vector<int64_t*>& values);

}  // namespace intel_npu::vm
