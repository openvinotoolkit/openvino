//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/print_utils.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"
#include "npu_bytecode_utils/span.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

const std::array<uint8_t, intel_npu::vm::MAGIC_NUMBER_SIZE>& intel_npu::vm::MagicNumber::value() const {
    return _value;
}

size_t intel_npu::vm::MagicNumber::getBinarySize() {
    return MAGIC_NUMBER_SIZE * sizeof(uint8_t);
}

void intel_npu::vm::MagicNumber::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, _value);
}

bool intel_npu::vm::MagicNumber::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, _value)) {
        NPU_VM_LOG_ERROR("Failed to parse magic number from buffer");
        return false;
    }
    if (!std::equal(MAGIC_NUMBER.begin(), MAGIC_NUMBER.end(), _value.begin(), _value.end())) {
        NPU_VM_LOG_ERROR("Magic number does not match expected value");
        return false;
    }
    return true;
}

void intel_npu::vm::MagicNumber::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    for (const auto& byte : _value) {
        std::cout << intel_npu::vm::toHex(byte) << " ";
    }
}
