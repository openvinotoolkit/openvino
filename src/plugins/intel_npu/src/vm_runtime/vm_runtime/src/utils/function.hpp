//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/managed_vector.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vm_export.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace intel_npu::vm {

struct NPU_VM_EXPORT FuncParamResType {
    intel_npu::vm::Type type{};
    int64_t typeSectionIndex{};
};

class NPU_VM_EXPORT Function {
    std::string _name;                               // Human-readable function name
    uint64_t _numGeneralRegisters;                   // Number of general-purpose registers used by the function
    bool _isEntrypoint;                              // True if this function is the bytecode entry point
    std::vector<FuncParamResType> _paramTypes;       // Types of the function parameters
    std::vector<FuncParamResType> _resultTypes;      // Types of the function result values
    ManagedVector<uint8_t> _body;                    // Raw instruction bytes representing the function body
    std::unordered_set<size_t> _instructionOffsets;  // Body-relative offsets where each instruction starts

public:
    /// Constructs a Function object with the given name, register count, entrypoint flag, parameter and result types,
    /// and instruction body. If copyBody is true, the Function object will make an owned copy of the body bytes;
    /// otherwise, it will reference the provided body bytes directly. The body bytes must remain valid for the lifetime
    /// of the Function object if copyBody is false.
    Function(std::string name, uint64_t numGeneralRegisters, bool isEntrypoint,
             std::vector<FuncParamResType> paramTypes, std::vector<FuncParamResType> resultTypes, Span<uint8_t> body,
             bool copyBody = false);

    /// Returns true if the Function object owns the body bytes (i.e., it made a copy of them), false otherwise
    bool ownsBody() const;

    /// Get the name of the function
    std::string getName() const;

    /// Get the number of general-purpose registers that the function uses
    uint64_t getNumGeneralRegisters() const;

    /// Returns true if this function is the bytecode entry point (i.e. the function that should be called externally to
    /// execute the bytecode)
    bool isEntrypoint() const;

    /// Get the parameter types of the function
    const std::vector<FuncParamResType>& getParamTypes() const;

    /// Get the result types of the function
    const std::vector<FuncParamResType>& getResultTypes() const;

    /// Get the raw instruction bytes representing the function body
    Span<const uint8_t> getBody() const;

    /// Walks the function body once and records the body-relative byte offset of every instruction start.
    /// Populates the internal instruction-offset table used by jump-target validation. Must be called before
    /// executing the function. Returns false if the body is malformed and cannot be fully decoded.
    bool parseInstructionOffsets();

    /// Returns true if the given body-relative byte offset corresponds to the start of an instruction.
    bool isValidInstructionOffset(size_t offset) const;
};

// Extracts a function parameter or result type from the type section given its index. The type is expected to be a
// primitive type (integer, float, opaque or buffer). The return value contains the extracted type and its original
// index in the type section
std::optional<FuncParamResType> NPU_VM_EXPORT extractFuncParamResType(intel_npu::vm::BytecodeReader& reader,
                                                                      size_t typeIndex);

}  // namespace intel_npu::vm
