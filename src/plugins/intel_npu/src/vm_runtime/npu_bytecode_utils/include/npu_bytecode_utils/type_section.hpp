//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/span.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

namespace intel_npu::vm {

class SectionHeaderTable;

enum class TypeCode : uint8_t {
    INTEGER = 0x01,
    FLOAT = 0x02,
    OPAQUE = 0x03,
    BUFFER = 0x04,
    FUNCTION = 0x05,
};

enum class FloatTypeFormat : uint8_t {
    IEEE754 = 0x00,
    BFloat = 0x01,
    TFloat = 0x02,
    E4M3 = 0x03,
    E5M2 = 0x04,
    E2M1 = 0x05,
    E8M0 = 0x06,
    NF4 = 0x07
};

struct IntegerType {
    uint8_t bitWidth;  // The bit width of the integer type (e.g. 32 for i32, 64 for i64)
    uint8_t isSigned;  // Whether the integer type is signed (1) or unsigned (0)
    TypeCode getTypeCode() const {
        return TypeCode::INTEGER;
    }

    uint8_t getBitWidth() const {
        return bitWidth;
    }

    uint8_t getSigned() const {
        return isSigned;
    }

    void appendTo(std::vector<uint8_t>& buffer) const;
    bool parseFrom(vm::Span<uint8_t>& buffer);
    void print(size_t indentLevel = 0) const;
};

struct FloatType {
    uint8_t bitWidth;        // The bit width of the float type (e.g. 32 for f32, 64 for f64)
    FloatTypeFormat format;  // The specific format of the float type (e.g. IEEE754, BFloat, etc.)

    TypeCode getTypeCode() const {
        return TypeCode::FLOAT;
    }

    uint8_t getBitWidth() const {
        return bitWidth;
    }

    void appendTo(std::vector<uint8_t>& buffer) const;
    bool parseFrom(vm::Span<uint8_t>& buffer);
    void print(size_t indentLevel = 0) const;
};

struct OpaqueType {
    uint8_t bitWidth;  // The bit width of the opaque type

    TypeCode getTypeCode() const {
        return TypeCode::OPAQUE;
    }

    uint8_t getBitWidth() const {
        return bitWidth;
    }

    void appendTo(std::vector<uint8_t>& buffer) const;
    bool parseFrom(vm::Span<uint8_t>& buffer);
    void print(size_t indentLevel = 0) const;
};

struct BufferType {
    using RankType = uint8_t;
    static constexpr int64_t MAX_RANK = std::numeric_limits<RankType>::max();

    uint64_t dataTypeIndex;        // Index into the type section for the data type of the buffer
    RankType rank;                 // The rank of the buffer (number of dimensions)
    std::vector<int64_t> shape;    // The shape of the buffer, with -1 representing dynamic dimensions
    std::vector<int64_t> strides;  // The strides of the buffer, with -1 representing dynamic strides

    TypeCode getTypeCode() const {
        return TypeCode::BUFFER;
    }

    void appendTo(std::vector<uint8_t>& buffer) const;
    bool parseFrom(vm::Span<uint8_t>& buffer);
    void print(size_t indentLevel = 0) const;
};

struct FunctionType {
    std::vector<uint64_t> paramTypeIndices;   // Indices into the type section for the parameter types of the function
    std::vector<uint64_t> resultTypeIndices;  // Indices into the type section for the result types of the function

    TypeCode getTypeCode() const {
        return TypeCode::FUNCTION;
    }

    void appendTo(std::vector<uint8_t>& buffer) const;
    bool parseFrom(vm::Span<uint8_t>& buffer);
    void print(size_t indentLevel = 0) const;
};

struct Type {
    // Note: FunctionType is currently excluded, as the current uses of Type only involve primitive types (such as for a
    // function's parameters or return types). If FunctionType is added in the future, it might be a good idea to keep
    // the differentiation between function and non-function types
    std::variant<IntegerType, FloatType, OpaqueType, BufferType> data{IntegerType{}};
};

TypeCode getTypeCode(const Type& type);

uint8_t getBitWidth(const intel_npu::vm::Type& type);

bool isTypeSigned(const intel_npu::vm::Type& type);

// Returns the rounded-up byte size for the type at `typeIndex` in the parsed type section.
// Returns 0 if the index is out of range or the type does not represent a primitive element type.
uint16_t lookupTypeByteSize(const std::vector<uint16_t>& typeByteSizes, int64_t typeIndex);

// Extracts rounded-up primitive byte sizes from the Type section. Non-primitive entries are returned as 0.
// Returns std::nullopt and prints an error if the type section layout is malformed.
std::optional<std::vector<uint16_t>> extractTypeByteSizes(const SectionHeaderTable& sectionHeaderTable,
                                                          const std::vector<Span<uint8_t>>& sections);

}  // namespace intel_npu::vm
