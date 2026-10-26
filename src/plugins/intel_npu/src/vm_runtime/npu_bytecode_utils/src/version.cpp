//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/version.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/print_utils.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"
#include "npu_bytecode_utils/span.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

size_t intel_npu::vm::Version::getBinarySize() {
    using VersionType = decltype(std::declval<intel_npu::vm::Version>().getFullVersion());
    return sizeof(VersionType);
}

void intel_npu::vm::Version::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, _version);
}

bool intel_npu::vm::Version::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, _version)) {
        NPU_VM_LOG_ERROR("Failed to parse version from buffer");
        return false;
    }
    return true;
}

void intel_npu::vm::Version::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << getMajor() << "." << getMinor() << "." << getPatch();
}
