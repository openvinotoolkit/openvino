//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/span.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace intel_npu::vm {

namespace details {

// Base class for section-specific metadata stored within a SectionHeader.
// Each section type (function, constant, etc.) carries its own layout description that is serialized immediately after
// the common header fields.
struct SectionInfo {
    SectionInfo() = default;
    SectionInfo(const SectionInfo&) = default;
    SectionInfo& operator=(const SectionInfo&) = default;
    SectionInfo(SectionInfo&&) = default;
    SectionInfo& operator=(SectionInfo&&) = default;
    virtual ~SectionInfo() = default;

    // Returns the total number of bytes this metadata occupies when serialized
    virtual size_t getBinarySize() const = 0;

    // Serializes the section-specific metadata and appends it to buffer
    virtual void appendTo(std::vector<uint8_t>& buffer) const = 0;

    // Deserializes function section metadata from the buffer.
    // Returns false and logs an error on malformed input.  The buffer is advanced past the consumed bytes.
    virtual bool parseFrom(vm::Span<uint8_t>& buffer) = 0;

    // Prints a human-readable representation of the metadata to stdout
    virtual void print(size_t indentLevel = 0) const = 0;
};

// Metadata for a function section. Describes all compiled functions including their names, type signatures, register
// requirements, and the byte range of each function body within the section payload.
struct FunctionSectionInfo : public SectionInfo {
    struct FunctionInfo {
        uint64_t nameIndex;            // Index into the string section for the function name
        uint64_t functionTypeIndex;    // Index into the type section for the function signature
        uint64_t numGeneralRegisters;  // Number of general registers used by the function
        uint64_t bodyOffset;           // Byte offset of the function body relative to the section start
        uint64_t bodySize;             // Size of the function body in bytes
    };

    uint64_t numFunctions{};                  // Total number of functions described in this section
    uint64_t entrypointFunctionIndex{};       // Index of the entry-point function within the function list
    std::vector<FunctionInfo> functionInfos;  // List of metadata entries for each function in the section

    size_t getBinarySize() const override;

    void appendTo(std::vector<uint8_t>& buffer) const override;

    bool parseFrom(vm::Span<uint8_t>& buffer) override;

    void print(size_t indentLevel = 0) const override;
};

// Metadata for data-oriented sections (constants, strings, types).
// Each entry is described by its offset and size within the section payload, allowing random access to individual data
// items.
struct DataSectionInfo : public SectionInfo {
    struct DataInfo {
        uint64_t offset;  // Byte offset of the data entry relative to the section start
        uint64_t size;    // Size of the data entry in bytes
    };

    uint64_t numData{};               // Total number of data entries in this section
    std::vector<DataInfo> dataInfos;  // List of metadata entries for each data item in the section

    size_t getBinarySize() const override;

    void appendTo(std::vector<uint8_t>& buffer) const override;

    bool parseFrom(vm::Span<uint8_t>& buffer) override;

    void print(size_t indentLevel = 0) const override;
};

}  // namespace details

/// Identifies the kind of payload a section carries.
/// @details The section type determines which SectionInfo subclass is used to describe the section layout.
enum class SectionType : uint8_t {
    FuncSection = 0x00,      // Contains compiled function bodies
    ConstantSection = 0x01,  // Contains constant data blobs (weights, biases, etc.)
    StringSection = 0x02,    // Contains null-terminated strings referenced by index
    KernelSection = 0x03,    // Contains kernel binaries referenced by index
    TypeSection = 0x04,      // Contains type descriptors (function signatures, buffer types, primitive types, etc.)
    MetadataSection = 0x05   // Contains serialized network/input/output metadata entries
};

std::string getSectionTypeString(SectionType type);

// Represents an entry in the section header table.
// Combines common fields (type, name, file offset, size) with a polymorphic
// SectionInfo pointer that holds section-type-specific metadata.
struct SectionHeader {
    SectionType type{};                          // Discriminator for the section payload kind
    uint64_t nameIndex{};                        // Index into the string section for the section name
    uint64_t offset{};                           // Absolute byte offset of the section payload in the file
    uint64_t size{};                             // Size of the section payload in bytes
    std::unique_ptr<details::SectionInfo> info;  // Section-type-specific descriptor

    SectionHeader() = default;
    SectionHeader(SectionType type, uint64_t nameIndex, uint64_t offset, uint64_t size,
                  std::unique_ptr<details::SectionInfo> info)
            : type(type), nameIndex(nameIndex), offset(offset), size(size), info(std::move(info)) {
    }
    SectionHeader(const SectionHeader&) = delete;
    SectionHeader& operator=(const SectionHeader&) = delete;
    SectionHeader(SectionHeader&&) = default;
    SectionHeader& operator=(SectionHeader&&) = default;
    ~SectionHeader() = default;

    // Returns the total serialized size of this header including its SectionInfo
    size_t getBinarySize() const;

    // Serializes the header fields and section-specific info into buffer
    void appendTo(std::vector<uint8_t>& buffer) const;

    // Deserializes a section header from buffer, including the appropriate SectionInfo subclass based on the parsed
    // section type.
    // Returns false and logs an error on failure.  The buffer is advanced past the consumed bytes.
    bool parseFrom(vm::Span<uint8_t>& buffer);

    // Prints a human-readable representation of the header and its metadata to stdout
    void print(size_t indentLevel = 0) const;
};

// Manages the collection of section headers that form the bytecode file's table of contents
class SectionHeaderTable {
    uint64_t _numSections = 0;
    std::vector<SectionHeader> _sectionHeaders;

public:
    SectionHeaderTable() = default;
    SectionHeaderTable(const SectionHeaderTable&) = delete;
    SectionHeaderTable& operator=(const SectionHeaderTable&) = delete;
    SectionHeaderTable(SectionHeaderTable&&) = default;
    SectionHeaderTable& operator=(SectionHeaderTable&&) = default;
    ~SectionHeaderTable() = default;

    // Returns a reference to the internal section header list
    const std::vector<SectionHeader>& getSectionHeaders() const;
    std::vector<SectionHeader>& getSectionHeaders();

    // Appends a section header to the table
    void addSectionHeader(SectionHeader header);

    // Returns the total serialized size of the table
    size_t getBinarySize() const;

    // Assigns absolute file offsets to each section's payload. Offsets are
    // calculated assuming sections are laid out immediately after the file
    // header in a fixed order: functions, constants, strings, types.
    void computeOffsets();

    // Serializes the section count and all headers into buffer.
    // Headers are written grouped by section type in a deterministic order:
    // functions, constants, strings, types.
    void appendTo(std::vector<uint8_t>& buffer) const;

    // Deserializes the section header table from buffer.
    // Returns false and logs an error if parsing fails. The buffer is advanced past the consumed bytes.
    bool parseFrom(vm::Span<uint8_t>& buffer);

    // Prints the entire section header table to stdout
    void print(size_t indentLevel = 0) const;
};

}  // namespace intel_npu::vm
