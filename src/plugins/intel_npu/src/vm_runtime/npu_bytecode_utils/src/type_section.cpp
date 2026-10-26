//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/type_section.hpp"
#include "npu_bytecode_utils/except.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/print_utils.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"
#include "npu_bytecode_utils/span.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <variant>
#include <vector>

namespace {

std::optional<uint16_t> extractPrimitiveTypeByteSize(intel_npu::vm::TypeCode typeCode,
                                                     intel_npu::vm::Span<uint8_t> typeData, size_t typeIndex) {
    switch (typeCode) {
    case intel_npu::vm::TypeCode::INTEGER: {
        intel_npu::vm::IntegerType type{};
        if (!type.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse integer type entry {}", typeIndex);
            return std::nullopt;
        }
        return (type.bitWidth + (CHAR_BIT - 1)) / CHAR_BIT;
    }
    case intel_npu::vm::TypeCode::FLOAT: {
        intel_npu::vm::FloatType type{};
        if (!type.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse float type entry {}", typeIndex);
            return std::nullopt;
        }
        return (type.bitWidth + (CHAR_BIT - 1)) / CHAR_BIT;
    }
    case intel_npu::vm::TypeCode::OPAQUE: {
        intel_npu::vm::OpaqueType type{};
        if (!type.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse opaque type entry {}", typeIndex);
            return std::nullopt;
        }
        return (type.bitWidth + (CHAR_BIT - 1)) / CHAR_BIT;
    }
    default:
        return 0;
    }
}

}  // namespace

void intel_npu::vm::IntegerType::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, static_cast<uint8_t>(TypeCode::INTEGER));
    appendValueTo(buffer, bitWidth);
    appendValueTo(buffer, isSigned);
}

bool intel_npu::vm::IntegerType::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, bitWidth)) {
        NPU_VM_LOG_ERROR("Failed to parse IntegerType bitWidth from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, isSigned)) {
        NPU_VM_LOG_ERROR("Failed to parse IntegerType isSigned from buffer");
        return false;
    }
    return true;
}

void intel_npu::vm::IntegerType::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << (isSigned ? "i" : "u") << static_cast<int>(bitWidth);
}

void intel_npu::vm::FloatType::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, static_cast<uint8_t>(TypeCode::FLOAT));
    appendValueTo(buffer, bitWidth);
    appendValueTo(buffer, static_cast<uint8_t>(format));
}

bool intel_npu::vm::FloatType::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, bitWidth)) {
        NPU_VM_LOG_ERROR("Failed to parse FloatType bitWidth from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, format)) {
        NPU_VM_LOG_ERROR("Failed to parse FloatType format from buffer");
        return false;
    }
    return true;
}

void intel_npu::vm::FloatType::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "float" << static_cast<int>(bitWidth);
    switch (format) {
    case FloatTypeFormat::IEEE754:
        std::cout << " (IEEE754)";
        break;
    case FloatTypeFormat::BFloat:
        std::cout << " (BFloat)";
        break;
    case FloatTypeFormat::TFloat:
        std::cout << " (TFloat)";
        break;
    case FloatTypeFormat::E4M3:
        std::cout << " (E4M3)";
        break;
    case FloatTypeFormat::E5M2:
        std::cout << " (E5M2)";
        break;
    case FloatTypeFormat::E2M1:
        std::cout << " (E2M1)";
        break;
    case FloatTypeFormat::E8M0:
        std::cout << " (E8M0)";
        break;
    case FloatTypeFormat::NF4:
        std::cout << " (NF4)";
        break;
    default:
        std::cout << " (Unknown format)";
        break;
    }
}

void intel_npu::vm::OpaqueType::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, static_cast<uint8_t>(TypeCode::OPAQUE));
    appendValueTo(buffer, bitWidth);
}

bool intel_npu::vm::OpaqueType::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, bitWidth)) {
        NPU_VM_LOG_ERROR("Failed to parse OpaqueType bitWidth from buffer");
        return false;
    }
    return true;
}

void intel_npu::vm::OpaqueType::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "opaque<" << static_cast<int>(bitWidth) << ">";
}

void intel_npu::vm::BufferType::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, static_cast<uint8_t>(TypeCode::BUFFER));
    appendValueTo(buffer, dataTypeIndex);
    appendValueTo(buffer, rank);
    for (const auto& dim : shape) {
        appendValueTo(buffer, dim);
    }
    for (const auto& stride : strides) {
        appendValueTo(buffer, stride);
    }
}

bool intel_npu::vm::BufferType::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, dataTypeIndex)) {
        NPU_VM_LOG_ERROR("Failed to parse BufferType dataTypeIndex from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, rank)) {
        NPU_VM_LOG_ERROR("Failed to parse BufferType rank from buffer");
        return false;
    }
    shape.resize(rank);
    strides.resize(rank);
    for (int64_t i = 0; i < rank; ++i) {
        if (!parseValueFrom(buffer, shape.at(i))) {
            NPU_VM_LOG_ERROR("Failed to parse BufferType shape dimension {}", i);
            return false;
        }
    }
    for (int64_t i = 0; i < rank; ++i) {
        if (!parseValueFrom(buffer, strides.at(i))) {
            NPU_VM_LOG_ERROR("Failed to parse BufferType stride dimension {}", i);
            return false;
        }
    }
    return true;
}

void intel_npu::vm::BufferType::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "buffer<typeIndex=" << dataTypeIndex << ", rank=" << static_cast<unsigned>(rank) << ", shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape.at(i);
        if (i < shape.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "], strides=[";
    for (size_t i = 0; i < strides.size(); ++i) {
        std::cout << strides.at(i);
        if (i < strides.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "]>";
}

void intel_npu::vm::FunctionType::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, static_cast<uint8_t>(TypeCode::FUNCTION));
    appendValueTo(buffer, static_cast<uint16_t>(paramTypeIndices.size()));
    for (const auto& paramTypeIndex : paramTypeIndices) {
        appendValueTo(buffer, paramTypeIndex);
    }
    appendValueTo(buffer, static_cast<uint16_t>(resultTypeIndices.size()));
    for (const auto& resultTypeIndex : resultTypeIndices) {
        appendValueTo(buffer, resultTypeIndex);
    }
}

bool intel_npu::vm::FunctionType::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    uint16_t numParams = 0;
    if (!parseValueFrom(buffer, numParams)) {
        NPU_VM_LOG_ERROR("Failed to parse FunctionType numParams from buffer");
        return false;
    }
    paramTypeIndices.resize(numParams);
    for (uint16_t i = 0; i < numParams; ++i) {
        if (!parseValueFrom(buffer, paramTypeIndices.at(i))) {
            NPU_VM_LOG_ERROR("Failed to parse FunctionType paramTypeIndex {}", i);
            return false;
        }
    }
    uint16_t numResults = 0;
    if (!parseValueFrom(buffer, numResults)) {
        NPU_VM_LOG_ERROR("Failed to parse FunctionType numResults from buffer");
        return false;
    }
    resultTypeIndices.resize(numResults);
    for (uint16_t i = 0; i < numResults; ++i) {
        if (!parseValueFrom(buffer, resultTypeIndices.at(i))) {
            NPU_VM_LOG_ERROR("Failed to parse FunctionType resultTypeIndex {}", i);
            return false;
        }
    }
    return true;
}

void intel_npu::vm::FunctionType::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "function<params=[";
    for (size_t i = 0; i < paramTypeIndices.size(); ++i) {
        std::cout << paramTypeIndices.at(i);
        if (i < paramTypeIndices.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "], results=[";
    for (size_t i = 0; i < resultTypeIndices.size(); ++i) {
        std::cout << resultTypeIndices.at(i);
        if (i < resultTypeIndices.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "]>";
}

intel_npu::vm::TypeCode intel_npu::vm::getTypeCode(const Type& type) {
    return std::visit(
            [](const auto& t) {
                return t.getTypeCode();
            },
            type.data);
}

uint8_t intel_npu::vm::getBitWidth(const intel_npu::vm::Type& type) {
    switch (getTypeCode(type)) {
    case TypeCode::INTEGER:
        return std::get<IntegerType>(type.data).bitWidth;
    case TypeCode::FLOAT:
        return std::get<FloatType>(type.data).bitWidth;
    case TypeCode::OPAQUE:
        return std::get<OpaqueType>(type.data).bitWidth;
    default:
        NPU_VM_THROW("Unsupported type code for getBitWidth: {}", static_cast<uint64_t>(getTypeCode(type)));
    }
}

bool intel_npu::vm::isTypeSigned(const intel_npu::vm::Type& type) {
    NPU_VM_THROW_UNLESS(getTypeCode(type) == TypeCode::INTEGER,
                        "isTypeSigned is only applicable to integer types, but got type code: {}",
                        static_cast<uint64_t>(getTypeCode(type)));
    return std::get<IntegerType>(type.data).isSigned;
}

uint16_t intel_npu::vm::lookupTypeByteSize(const std::vector<uint16_t>& typeByteSizes, int64_t typeIndex) {
    if (typeIndex < 0 || static_cast<size_t>(typeIndex) >= typeByteSizes.size()) {
        return 0;
    }
    return typeByteSizes.at(static_cast<size_t>(typeIndex));
}

std::optional<std::vector<uint16_t>> intel_npu::vm::extractTypeByteSizes(
        const intel_npu::vm::SectionHeaderTable& sectionHeaderTable,
        const std::vector<intel_npu::vm::Span<uint8_t>>& sections) {
    const auto& sectionHeaders = sectionHeaderTable.getSectionHeaders();
    const auto numTypeSections = std::count_if(sectionHeaders.begin(), sectionHeaders.end(), [](const auto& header) {
        return header.type == intel_npu::vm::SectionType::TypeSection;
    });
    if (numTypeSections > 1) {
        NPU_VM_LOG_ERROR("Expected at most one Type section, but found {}", numTypeSections);
        return std::nullopt;
    }

    std::vector<uint16_t> typeByteSizes;
    for (size_t headerIdx = 0; headerIdx < sectionHeaders.size(); ++headerIdx) {
        const auto& header = sectionHeaders.at(headerIdx);
        if (header.type != intel_npu::vm::SectionType::TypeSection) {
            continue;
        }
        auto typeSectionInfo = dynamic_cast<intel_npu::vm::details::DataSectionInfo*>(header.info.get());
        if (typeSectionInfo == nullptr) {
            NPU_VM_LOG_ERROR("Type section header does not contain data section info");
            return std::nullopt;
        }
        if (headerIdx >= sections.size()) {
            NPU_VM_LOG_ERROR("Type section header has no payload");
            return std::nullopt;
        }
        const auto& section = sections.at(headerIdx);
        for (size_t typeIndex = 0; typeIndex < typeSectionInfo->dataInfos.size(); ++typeIndex) {
            const auto& dataInfo = typeSectionInfo->dataInfos.at(typeIndex);
            if (dataInfo.size == 0 || dataInfo.offset > section.size() ||
                dataInfo.size > section.size() - dataInfo.offset) {
                NPU_VM_LOG_ERROR("Type entry {} exceeds section bounds", typeIndex);
                return std::nullopt;
            }

            const auto typeSpan = section.subspan(dataInfo.offset, dataInfo.size);
            const auto typeCode = static_cast<intel_npu::vm::TypeCode>(typeSpan.at(0));
            auto typeData = typeSpan.subspan(sizeof(uint8_t), typeSpan.size() - 1);
            auto byteSize = extractPrimitiveTypeByteSize(typeCode, typeData, typeIndex);
            if (!byteSize.has_value()) {
                return std::nullopt;
            }
            typeByteSizes.push_back(byteSize.value());
        }
    }

    return typeByteSizes;
}
