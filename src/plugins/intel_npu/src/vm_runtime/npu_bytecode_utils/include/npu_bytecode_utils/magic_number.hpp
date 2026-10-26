//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/span.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace intel_npu::vm {

constexpr uint8_t MAGIC_NUMBER_SIZE = 8;
constexpr std::array<uint8_t, MAGIC_NUMBER_SIZE> MAGIC_NUMBER = {0x4E, 0x50, 0x55, 0x42,
                                                                 0x79, 0x74, 0x65, 0x00};  // 'NPUByte\0'

class MagicNumber {
    std::array<uint8_t, MAGIC_NUMBER_SIZE> _value;

public:
    MagicNumber() = default;
    MagicNumber(const std::array<uint8_t, MAGIC_NUMBER_SIZE>& value): _value(value) {
    }

    const std::array<uint8_t, MAGIC_NUMBER_SIZE>& value() const;

    // Returns the total number of bytes used to represent the magic number
    static size_t getBinarySize();

    // Appends the magic number bytes to the provided buffer
    void appendTo(std::vector<uint8_t>& buffer) const;

    // Parses the magic number from the given buffer and populates the value field.
    // Returns false and logs an error on malformed input. The buffer is advanced past the consumed bytes on success.
    bool parseFrom(vm::Span<uint8_t>& buffer);

    // Prints the magic number in hexadecimal format to stdout
    void print(size_t indentLevel = 0) const;
};

}  // namespace intel_npu::vm
