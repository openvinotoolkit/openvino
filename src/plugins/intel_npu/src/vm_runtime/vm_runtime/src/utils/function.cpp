//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "function.hpp"
#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/type_section.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

intel_npu::vm::Function::Function(std::string name, uint64_t numGeneralRegisters, bool isEntrypoint,
                                  std::vector<FuncParamResType> paramTypes, std::vector<FuncParamResType> resultTypes,
                                  intel_npu::vm::Span<uint8_t> body, bool copyBody)
        : _name(std::move(name)),
          _numGeneralRegisters(numGeneralRegisters),
          _isEntrypoint(isEntrypoint),
          _paramTypes(std::move(paramTypes)),
          _resultTypes(std::move(resultTypes)),
          _body(body, copyBody) {
}

bool intel_npu::vm::Function::ownsBody() const {
    return _body.isOwned();
}

std::string intel_npu::vm::Function::getName() const {
    return _name;
}

uint64_t intel_npu::vm::Function::getNumGeneralRegisters() const {
    return _numGeneralRegisters;
}

bool intel_npu::vm::Function::isEntrypoint() const {
    return _isEntrypoint;
}

const std::vector<intel_npu::vm::FuncParamResType>& intel_npu::vm::Function::getParamTypes() const {
    return _paramTypes;
}

const std::vector<intel_npu::vm::FuncParamResType>& intel_npu::vm::Function::getResultTypes() const {
    return _resultTypes;
}

intel_npu::vm::Span<const uint8_t> intel_npu::vm::Function::getBody() const {
    return _body.get();
}

bool intel_npu::vm::Function::parseInstructionOffsets() {
    _instructionOffsets.clear();
    const auto body = _body.get();
    const auto begin = body.begin();
    const auto totalSize = body.size();

    size_t offset = 0;
    while (offset < totalSize) {
        const auto remaining = totalSize - offset;
        if (remaining < intel_npu::vm::OPCODE_SIZE) {
            NPU_VM_LOG_ERROR("Function '{}' body is malformed: cannot read opcode at offset {}", _name, offset);
            return false;
        }
        const auto* instructionBegin = std::next(begin, static_cast<std::ptrdiff_t>(offset));
        const auto opcode = intel_npu::vm::getOpcode(instructionBegin);

        const auto instructionSize = intel_npu::vm::getInstructionSize(opcode, instructionBegin, remaining);
        if (!instructionSize.has_value() || instructionSize.value() == 0) {
            NPU_VM_LOG_ERROR(
                    "Function '{}' body is malformed: failed to determine instruction size at offset {} (unknown "
                    "opcode {} or truncated instruction)",
                    _name, offset, static_cast<uint16_t>(opcode));
            return false;
        }
        if (remaining < instructionSize.value()) {
            NPU_VM_LOG_ERROR("Function '{}' body is malformed: instruction at offset {} exceeds body bounds", _name,
                             offset);
            return false;
        }

        _instructionOffsets.insert(offset);
        offset += instructionSize.value();
    }
    return true;
}

bool intel_npu::vm::Function::isValidInstructionOffset(size_t offset) const {
    return _instructionOffsets.count(offset) != 0;
}

std::optional<intel_npu::vm::FuncParamResType> intel_npu::vm::extractFuncParamResType(
        intel_npu::vm::BytecodeReader& reader, size_t typeIndex) {
    auto typeEntry = reader.getDataSectionEntry(intel_npu::vm::SectionType::TypeSection, typeIndex);
    if (typeEntry.empty()) {
        NPU_VM_LOG_ERROR("Failed to retrieve type entry for type index {}", typeIndex);
        return std::nullopt;
    }

    const auto typeCode = static_cast<intel_npu::vm::TypeCode>(typeEntry.at(0));
    if (typeCode == intel_npu::vm::TypeCode::INTEGER) {
        intel_npu::vm::IntegerType intType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!intType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse IntegerType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{intType}, static_cast<int64_t>(typeIndex)};
    } else if (typeCode == intel_npu::vm::TypeCode::FLOAT) {
        intel_npu::vm::FloatType floatType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!floatType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse FloatType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{floatType}, static_cast<int64_t>(typeIndex)};
    } else if (typeCode == intel_npu::vm::TypeCode::OPAQUE) {
        intel_npu::vm::OpaqueType opaqueType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!opaqueType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse OpaqueType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{opaqueType}, static_cast<int64_t>(typeIndex)};
    } else if (typeCode == intel_npu::vm::TypeCode::BUFFER) {
        intel_npu::vm::BufferType bufferType{};
        typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
        if (!bufferType.parseFrom(typeEntry)) {
            NPU_VM_LOG_ERROR("Failed to parse BufferType from type entry for type index {}", typeIndex);
            return std::nullopt;
        }
        return FuncParamResType{intel_npu::vm::Type{bufferType}, static_cast<int64_t>(typeIndex)};
    } else {
        NPU_VM_LOG_ERROR("Unsupported type code {} for type index {}", static_cast<uint64_t>(typeCode), typeIndex);
        return std::nullopt;
    }
}
