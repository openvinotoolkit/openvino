//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/span.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace intel_npu::vm {

template <typename T>
void appendValueTo(std::vector<uint8_t>& buffer, const T& value) {
    std::array<uint8_t, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    buffer.insert(buffer.end(), bytes.begin(), bytes.end());
}

template <typename T>
bool parseValueFrom(vm::Span<uint8_t>& buffer, T& value) {
    if (buffer.begin() == nullptr) {
        return false;
    }
    if (buffer.size() < sizeof(T)) {
        return false;
    }
    memcpy(&value, buffer.begin(), sizeof(T));
    buffer = buffer.subspan(sizeof(T));
    return true;
}

}  // namespace intel_npu::vm
