//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/print_utils.hpp"
#include "npu_bytecode_utils/serialization_utils.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/version.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

size_t intel_npu::vm::details::FunctionSectionInfo::getBinarySize() const {
    const auto functionInfoSize = sizeof(FunctionInfo::nameIndex) + sizeof(FunctionInfo::functionTypeIndex) +
                                  sizeof(FunctionInfo::numGeneralRegisters) + sizeof(FunctionInfo::bodyOffset) +
                                  sizeof(FunctionInfo::bodySize);
    return sizeof(numFunctions) + sizeof(entrypointFunctionIndex) + functionInfos.size() * functionInfoSize;
}

void intel_npu::vm::details::FunctionSectionInfo::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, numFunctions);
    appendValueTo(buffer, entrypointFunctionIndex);
    for (const auto& functionInfo : functionInfos) {
        appendValueTo(buffer, functionInfo.nameIndex);
        appendValueTo(buffer, functionInfo.functionTypeIndex);
        appendValueTo(buffer, functionInfo.numGeneralRegisters);
        appendValueTo(buffer, functionInfo.bodyOffset);
        appendValueTo(buffer, functionInfo.bodySize);
    }
}

bool intel_npu::vm::details::FunctionSectionInfo::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, numFunctions)) {
        NPU_VM_LOG_ERROR("Failed to parse numFunctions from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, entrypointFunctionIndex)) {
        NPU_VM_LOG_ERROR("Failed to parse entrypointFunctionIndex from buffer");
        return false;
    }
    for (uint64_t i = 0; i < numFunctions; ++i) {
        FunctionSectionInfo::FunctionInfo functionInfo{};
        if (!parseValueFrom(buffer, functionInfo.nameIndex)) {
            NPU_VM_LOG_ERROR("Failed to parse nameIndex from buffer, for function {}", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.functionTypeIndex)) {
            NPU_VM_LOG_ERROR("Failed to parse functionTypeIndex from buffer, for function {}", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.numGeneralRegisters)) {
            NPU_VM_LOG_ERROR("Failed to parse numGeneralRegisters from buffer, for function {}", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.bodyOffset)) {
            NPU_VM_LOG_ERROR("Failed to parse bodyOffset from buffer, for function {}", i);
            return false;
        }
        if (!parseValueFrom(buffer, functionInfo.bodySize)) {
            NPU_VM_LOG_ERROR("Failed to parse bodySize from buffer {} {}, for function {}", buffer.begin(),
                             buffer.size(), i);
            return false;
        }
        functionInfos.push_back(functionInfo);
    }
    return true;
}

void intel_npu::vm::details::FunctionSectionInfo::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Number of functions: " << numFunctions << ", entrypoint function index: " << entrypointFunctionIndex
              << std::endl;
    for (const auto& functionInfo : functionInfos) {
        intel_npu::vm::printIndent(indentLevel + 1);
        // TODO: Replace name index and type index with actual entries from string / type sections
        std::cout << "Name index: " << functionInfo.nameIndex
                  << ", function type index: " << functionInfo.functionTypeIndex
                  << ", num general registers: " << functionInfo.numGeneralRegisters
                  << ", body offset: " << functionInfo.bodyOffset << ", body size: " << functionInfo.bodySize
                  << std::endl;
    }
}

size_t intel_npu::vm::details::DataSectionInfo::getBinarySize() const {
    const auto dataInfoSize = sizeof(DataInfo::offset) + sizeof(DataInfo::size);
    return sizeof(numData) + dataInfos.size() * dataInfoSize;
}

void intel_npu::vm::details::DataSectionInfo::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, numData);
    for (const auto& dataInfo : dataInfos) {
        appendValueTo(buffer, dataInfo.offset);
        appendValueTo(buffer, dataInfo.size);
    }
}

bool intel_npu::vm::details::DataSectionInfo::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, numData)) {
        NPU_VM_LOG_ERROR("Failed to parse numData from buffer");
        return false;
    }
    for (uint64_t i = 0; i < numData; ++i) {
        DataSectionInfo::DataInfo dataInfo{};
        if (!parseValueFrom(buffer, dataInfo.offset)) {
            NPU_VM_LOG_ERROR("Failed to parse offset from buffer, for data {}", i);
            return false;
        }
        if (!parseValueFrom(buffer, dataInfo.size)) {
            NPU_VM_LOG_ERROR("Failed to parse size from buffer, for data {}", i);
            return false;
        }
        dataInfos.push_back(dataInfo);
    }
    return true;
}

void intel_npu::vm::details::DataSectionInfo::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Number of entries: " << numData << std::endl;
    for (uint64_t i = 0; i < dataInfos.size(); ++i) {
        const auto& dataInfo = dataInfos.at(i);
        intel_npu::vm::printIndent(indentLevel + 1);
        std::cout << "Entry " << i << " offset: " << dataInfo.offset << ", size: " << dataInfo.size << std::endl;
    }
}

std::string intel_npu::vm::getSectionTypeString(intel_npu::vm::SectionType type) {
    switch (type) {
    case SectionType::FuncSection:
        return "Function";
    case SectionType::ConstantSection:
        return "Constant";
    case SectionType::StringSection:
        return "String";
    case SectionType::KernelSection:
        return "Kernel";
    case SectionType::TypeSection:
        return "Type";
    case SectionType::MetadataSection:
        return "Metadata";
    default:
        return "Unknown";
    }
}

size_t intel_npu::vm::SectionHeader::getBinarySize() const {
    return sizeof(type) + sizeof(nameIndex) + sizeof(offset) + sizeof(size) + info->getBinarySize();
}

void intel_npu::vm::SectionHeader::appendTo(std::vector<uint8_t>& buffer) const {
    appendValueTo(buffer, static_cast<uint8_t>(type));
    appendValueTo(buffer, nameIndex);
    appendValueTo(buffer, offset);
    appendValueTo(buffer, size);
    info->appendTo(buffer);
}

bool intel_npu::vm::SectionHeader::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    if (!parseValueFrom(buffer, type)) {
        NPU_VM_LOG_ERROR("Failed to parse type from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, nameIndex)) {
        NPU_VM_LOG_ERROR("Failed to parse nameIndex from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, offset)) {
        NPU_VM_LOG_ERROR("Failed to parse offset from buffer");
        return false;
    }
    if (!parseValueFrom(buffer, size)) {
        NPU_VM_LOG_ERROR("Failed to parse size from buffer");
        return false;
    }
    if (type == SectionType::FuncSection) {
        auto functionSectionInfo = details::FunctionSectionInfo{};
        if (!functionSectionInfo.parseFrom(buffer)) {
            NPU_VM_LOG_ERROR("Failed to parse function section info from buffer");
            return false;
        }
        info = std::make_unique<details::FunctionSectionInfo>(functionSectionInfo);
    } else if (type == SectionType::ConstantSection || type == SectionType::KernelSection ||
               type == SectionType::StringSection || type == SectionType::TypeSection ||
               type == SectionType::MetadataSection) {
        auto dataSectionInfo = details::DataSectionInfo{};
        if (!dataSectionInfo.parseFrom(buffer)) {
            NPU_VM_LOG_ERROR("Failed to parse data section info from buffer");
            return false;
        }
        info = std::make_unique<details::DataSectionInfo>(dataSectionInfo);
    } else {
        NPU_VM_LOG_ERROR("Unknown section type {}", static_cast<uint8_t>(type));
        return false;
    }
    return true;
}

void intel_npu::vm::SectionHeader::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Section type: " << getSectionTypeString(type) << ", name index: " << nameIndex
              << ", offset: " << offset << ", size: " << size << std::endl;
    info->print(indentLevel + 1);
}

const std::vector<intel_npu::vm::SectionHeader>& intel_npu::vm::SectionHeaderTable::getSectionHeaders() const {
    return _sectionHeaders;
}

std::vector<intel_npu::vm::SectionHeader>& intel_npu::vm::SectionHeaderTable::getSectionHeaders() {
    return _sectionHeaders;
}

void intel_npu::vm::SectionHeaderTable::addSectionHeader(SectionHeader header) {
    _sectionHeaders.push_back(std::move(header));
    ++_numSections;
}

size_t intel_npu::vm::SectionHeaderTable::getBinarySize() const {
    size_t size = sizeof(_numSections);
    for (const auto& header : _sectionHeaders) {
        size += header.getBinarySize();
    }
    return size;
}

void intel_npu::vm::SectionHeaderTable::computeOffsets() {
    // Section payloads start immediately after the file header, which consists of magic number, version, and
    // the section header table itself
    uint64_t currentOffset = MagicNumber::getBinarySize() + Version::getBinarySize() + getBinarySize();
    // Assign offsets in a deterministic order so that the file layout is predictable: functions first, then constants,
    // strings, and finally types
    const auto updateOffsets = [&](SectionType sectionType) {
        for (auto& header : _sectionHeaders) {
            if (header.type == sectionType) {
                header.offset = currentOffset;
                currentOffset += header.size;
            }
        }
    };
    updateOffsets(SectionType::FuncSection);
    updateOffsets(SectionType::ConstantSection);
    updateOffsets(SectionType::KernelSection);
    updateOffsets(SectionType::StringSection);
    updateOffsets(SectionType::TypeSection);
    updateOffsets(SectionType::MetadataSection);
}

void intel_npu::vm::SectionHeaderTable::appendTo(std::vector<uint8_t>& buffer) const {
    // Serialize headers grouped by section type to match the offset assignment order used in computeOffsets()
    const auto appendSection = [&](SectionType sectionType) {
        for (const auto& header : _sectionHeaders) {
            if (header.type == sectionType) {
                header.appendTo(buffer);
            }
        }
    };
    appendValueTo(buffer, _numSections);
    appendSection(SectionType::FuncSection);
    appendSection(SectionType::ConstantSection);
    appendSection(SectionType::KernelSection);
    appendSection(SectionType::StringSection);
    appendSection(SectionType::TypeSection);
    appendSection(SectionType::MetadataSection);
}

bool intel_npu::vm::SectionHeaderTable::parseFrom(intel_npu::vm::Span<uint8_t>& buffer) {
    uint64_t fileNumSections = 0;
    if (!parseValueFrom(buffer, fileNumSections)) {
        NPU_VM_LOG_ERROR("Failed to parse numSections from buffer");
        return false;
    }
    for (uint64_t i = 0; i < fileNumSections; ++i) {
        SectionHeader header{};
        if (!header.parseFrom(buffer)) {
            NPU_VM_LOG_ERROR("Failed to parse section header {} from buffer", i);
            return false;
        }
        addSectionHeader(std::move(header));
    }
    return true;
}

void intel_npu::vm::SectionHeaderTable::print(size_t indentLevel) const {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Number of sections: " << _numSections << std::endl;
    for (const auto& header : _sectionHeaders) {
        header.print(indentLevel + 1);
    }
}
