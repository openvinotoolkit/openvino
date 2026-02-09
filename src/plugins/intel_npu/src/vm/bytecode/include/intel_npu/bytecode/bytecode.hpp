//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <istream>
#include <vector>

#include <cstdint>
#include "vpux_headers/metadata.hpp"

namespace vpux {

constexpr uint64_t MaxDimCount = 5;

struct MemRefDesc {
    void* data;
    int64_t offset;
    int64_t dimCount;
    int64_t sizes[MaxDimCount];
    int64_t strides[MaxDimCount];
    int64_t elementByteSize;
};

struct BinWriter {
    std::ostream& out;

    explicit BinWriter(std::ostream& output): out(output) {
    }

    template <typename T>
    void write(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        out.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }

    void write(const std::string& s) {
        uint64_t n = s.size();
        write(n);
        out.write(s.data(), n);
    }

    template <typename T>
    void write(const std::vector<T>& v) {
        uint64_t n = v.size();
        write(n);
        for (const auto& e : v) {
            write(e);
        }
    }

    template <typename K, typename V>
    void write(const std::unordered_map<K, V>& m) {
        uint64_t n = m.size();
        write(n);
        for (const auto& [key, value] : m) {
            write(key);
            write(value);
        }
    }
};

struct BinReader {
    std::istream& in;
    explicit BinReader(std::istream& input): in(input) {
    }

    template <typename T>
    void read(T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        in.read(reinterpret_cast<char*>(&v), sizeof(T));
    }

    void read(std::string& s) {
        uint64_t n;
        read(n);
        s.resize(n);
        in.read(s.data(), n);
    }

    template <typename T>
    void read(std::vector<T>& v) {
        uint64_t n;
        read(n);
        v.resize(n);
        for (auto& e : v) {
            read(e);
        }
    }

    template <typename K, typename V>
    void read(std::unordered_map<K, V>& m) {
        uint64_t n;
        read(n);
        for (uint64_t i = 0; i < n; ++i) {
            K key;
            V value;
            read(key);
            read(value);
            m[key] = value;
        }
    }
};

struct Buffer {
    void* data;
    size_t size;
};

struct Kernel {
    std::vector<uint8_t> binaryData;
    std::string symbolName;

    void serialize(BinWriter& w) const {
        w.write(binaryData);
        w.write(symbolName);
    }

    void deserialize(BinReader& r) {
        r.read(binaryData);
        r.read(symbolName);
    }
};

struct KernelDesc {
    uint32_t kernelId;
    std::vector<uint32_t> argSlots;
};

enum class OffsetKind : uint8_t { Imm = 0, Reg = 1 };

struct OffsetSpec {
    OffsetKind kind = OffsetKind::Imm;
    int64_t imm = 0;   // used if kind==Imm
    uint32_t reg = 0;  // i64 reg id if kind==Reg
};

struct SubviewTemplate {
    uint8_t rank = 0;                 // number of dims
    OffsetSpec offsets[4];            // per-dim offset (elements)
    int64_t sizes[4] = {0, 0, 0, 0};  // result sizes (elements)
                                      // step assumed 1 for prototype; can add step[] later
};
enum class CmpPred : int64_t { EQ, NE, LT, LE, GT, GE };

enum class OpCode : uint8_t {
    I64_CONST = 0,
    I64_MOV = 1,
    I64_ADD = 2,
    I64_SUB = 3,
    I64_MUL = 4,
    I64_DIV = 5,
    I64_REM = 6,
    I64_MIN = 7,
    I64_SELECT = 8,
    I64_CMP = 9,  // bool = (a < b)
    JMP = 10,
    BR_IF = 11,  // if boolReg true -> pcTrue else pcFalse
    CALL = 12,
    RET = 13,
    RUN_STAGE = 14,
    CALL_KERNEL = 15,
    MEMREF_SUBVIEW = 16,
    I64_MEMREF_DIM = 17,  // i64dst = memrefSlot.sizes[dimIndex]
    SUBMIT = 18,
    ASSERT = 19,
};

struct Instruction {
    OpCode op{};
    uint32_t a = 0, b = 0, c = 0, d = 0;  // dst, src1, src2, src3 in case of multi-operand ops

    int64_t imm = 0;
};

struct Function {
    std::string name;
    uint32_t numI64 = 0;
    uint32_t numMem = 0;
    std::vector<Instruction> code;

    void serialize(BinWriter& w) const {
        w.write(name);
        w.write(numI64);
        w.write(numMem);
        w.write(code);
    }

    void deserialize(BinReader& r) {
        r.read(name);
        r.read(numI64);
        r.read(numMem);
        r.read(code);
    }
};

struct ExecutionPlan {
    std::vector<Function> functions;
    uint32_t entryFuncId = 0;

    std::vector<SubviewTemplate> subviews;
    std::vector<MemRefDesc> operands;
    // std::vector<MemRefDesc> buffers;

    // Buffer scratchBuffer;
    // TODO: is it cross-platform safe to serialize unordered_map?
    std::unordered_map<uint32_t, Kernel> kernelMap;
    std::vector<KernelDesc> kernelDescs;

    // TODO: should we depend on elf library here?
    elf::NetworkMetadata networkMetadata;
    std::vector<uint8_t> binaryMetadata;

    void serialize(std::ostream& output);
    void deserialize(std::istream& input);
    void print();
};

}  // namespace vpux
