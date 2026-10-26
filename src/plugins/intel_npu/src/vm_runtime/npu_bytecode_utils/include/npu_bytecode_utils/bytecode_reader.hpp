//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "npu_bytecode_utils/version.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace intel_npu::vm {

// Deserializes a bytecode binary into its constituent parts: file header and section payloads
class BytecodeReader {
    Span<uint8_t> _bytecode;

    MagicNumber _magicNumber{};              // File format identifier parsed from the bytecode header
    Version _version{};                      // Bytecode format version (major.minor.patch)
    SectionHeaderTable _sectionHeaderTable;  // Table of contents describing all sections in the file
    std::vector<Span<uint8_t>> _sections;    // Raw payload bytes for each section, indexed in header order

    // Oldest bytecode version the VM still supports (backward compat floor)
    static Version getMinSupportedVersion();
    // Current VM version: the newest bytecode format the VM implements
    static Version getMaxSupportedVersion();

    // Parses the file header fields (magic number, version, section header table) from the beginning of the bytecode
    // buffer. Returns false on malformed input.
    bool parseFileHeader();

    // Extracts raw section payloads using offsets and sizes from the previously
    // parsed section header table. Must be called after parseFileHeader().
    // Returns false if any section exceeds the bytecode buffer bounds.
    bool parseSections();

    void printFunctionSection(const SectionHeader& sectionHeader, Span<uint8_t> sectionContent, size_t sectionIdx,
                              size_t indentLevel);
    void printDataSection(const SectionHeader& sectionHeader, Span<uint8_t> sectionContent, size_t sectionIdx,
                          SectionType sectionType, bool printFull, size_t indentLevel);

public:
    // Constructs a reader for the given bytecode buffer. The buffer must remain valid for the lifetime of the reader
    BytecodeReader(Span<uint8_t> bytecode): _bytecode(bytecode), _sectionHeaderTable() {
    }

    // Returns a reference to the parsed section header table. Valid only after a successful parseFile() call.
    SectionHeaderTable& getSectionHeaderTable();

    // Returns the extracted section payloads as a list of byte vectors,
    // ordered to match the section headers. Valid only after a successful parseFile() call.
    const std::vector<Span<uint8_t>>& getSections() const;

    Span<uint8_t> getDataSectionEntry(SectionType sectionType, size_t entryIndex) const;

    /// Retrieves a string from the string section given its index, Valid only after a successful parseFile() call.
    /// @param stringIndex The index of the desired string in the string section
    /// @return The string value if found, or std::nullopt otherwise
    std::optional<std::string> getString(size_t stringIndex) const;

    std::optional<FunctionType> getFunctionType(size_t typeIndex) const;

    // Checks whether the parsed bytecode version is compatible with this VM
    static bool isVersionSupported(Span<uint8_t> bytecode, Version minVersion = getMinSupportedVersion(),
                                   Version maxVersion = getMaxSupportedVersion());

    // Checks whether the parsed bytecode version is compatible with this VM
    bool isVersionSupported() const;

    // Parses the entire bytecode file: header followed by section payloads.
    // Returns false and logs an error if any stage of parsing fails.
    bool parseFile();

    // Prints the parsed file header to stdout
    void printFileHeader(size_t indentLevel = 0);

    /// Prints the entire parsed file to stdout
    /// @param printFull If true, also prints the content of binary sections such constants or kernels
    /// @param indentLevel The indentation level for pretty-printing nested structures
    void printFile(bool printFull = true, size_t indentLevel = 0);
};

}  // namespace intel_npu::vm
