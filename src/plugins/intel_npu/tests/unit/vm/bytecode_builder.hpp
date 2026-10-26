//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "function.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "npu_bytecode_utils/version.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace utils {

// Encodes bytecode instructions and manages bytecode file generation
class BytecodeBuilder {
private:
    // Context for a function with its definition and name / type indices into the string and type sections
    struct FunctionContext {
        intel_npu::vm::Function definition;
        uint64_t nameIndex{};          // Pre-interned index into the string section
        uint64_t functionTypeIndex{};  // Pre-interned index into the type section
    };

    // Function list order defines function indices referenced by CALL instructions
    std::deque<FunctionContext> _functions;

    // Bytecode version serialized into the output binary
    intel_npu::vm::Version _version{1, 0, 0};

    // Incrementally built string section state
    std::vector<uint8_t> _stringSectionPayload;
    intel_npu::vm::details::DataSectionInfo _stringSectionInfo;

    // Incrementally built type section state
    std::vector<uint8_t> _typeSectionPayload;
    intel_npu::vm::details::DataSectionInfo _typeSectionInfo;

    // Interns a string into the string section, returning its index
    uint64_t internString(std::string_view str) {
        // String section entries are null-terminated
        const auto makeNullTerminatedString = [](std::string_view str) -> std::vector<uint8_t> {
            std::vector<uint8_t> result(str.begin(), str.end());
            result.push_back('\0');
            return result;
        };

        const auto index = static_cast<uint64_t>(_stringSectionInfo.dataInfos.size());
        const auto bytes = makeNullTerminatedString(str);
        intel_npu::vm::details::DataSectionInfo::DataInfo dataInfo{};
        dataInfo.offset = _stringSectionPayload.size();
        dataInfo.size = bytes.size();
        _stringSectionPayload.insert(_stringSectionPayload.end(), bytes.begin(), bytes.end());
        _stringSectionInfo.dataInfos.push_back(dataInfo);
        _stringSectionInfo.numData = _stringSectionInfo.dataInfos.size();
        return index;
    }

    // Interns a serialized type blob into the type section, returning its index
    uint64_t internTypeBytes(const std::vector<uint8_t>& serialized) {
        const auto index = static_cast<uint64_t>(_typeSectionInfo.dataInfos.size());
        intel_npu::vm::details::DataSectionInfo::DataInfo dataInfo{};
        dataInfo.offset = _typeSectionPayload.size();
        dataInfo.size = serialized.size();
        _typeSectionPayload.insert(_typeSectionPayload.end(), serialized.begin(), serialized.end());
        _typeSectionInfo.dataInfos.push_back(dataInfo);
        _typeSectionInfo.numData = _typeSectionInfo.dataInfos.size();
        return index;
    }

    // Interns a primitive (integer, float, opaque, buffer) type, returning its type section index
    uint64_t internNonFunctionType(const intel_npu::vm::Type& type) {
        std::vector<uint8_t> entry;
        switch (intel_npu::vm::getTypeCode(type)) {
        case intel_npu::vm::TypeCode::INTEGER:
            std::get<intel_npu::vm::IntegerType>(type.data).appendTo(entry);
            break;
        case intel_npu::vm::TypeCode::FLOAT:
            std::get<intel_npu::vm::FloatType>(type.data).appendTo(entry);
            break;
        case intel_npu::vm::TypeCode::OPAQUE:
            std::get<intel_npu::vm::OpaqueType>(type.data).appendTo(entry);
            break;
        case intel_npu::vm::TypeCode::BUFFER:
            std::get<intel_npu::vm::BufferType>(type.data).appendTo(entry);
            break;
        default:
            throw std::runtime_error("Unsupported primitive type code: " +
                                     std::to_string(static_cast<uint64_t>(intel_npu::vm::getTypeCode(type))));
        }
        return internTypeBytes(entry);
    }

    // Interns a function type signature, returning its type section index
    uint64_t internFunctionType(const intel_npu::vm::FunctionType& funcType) {
        std::vector<uint8_t> entry;
        funcType.appendTo(entry);
        return internTypeBytes(entry);
    }

public:
    // Builds and encodes a single function's instruction body
    class FunctionBuilder {
    private:
        std::string _name;
        int64_t _numGeneralRegisters;
        std::vector<intel_npu::vm::Type> _paramTypes;
        std::vector<intel_npu::vm::Type> _resultTypes;
        bool _isEntrypoint;
        std::vector<uint8_t> _body;

        friend BytecodeBuilder;

        // Appends a value to the function body
        template <typename T>
        void append(T value) {
            std::array<uint8_t, sizeof(T)> buf{};
            std::memcpy(buf.data(), &value, sizeof(T));
            _body.insert(_body.end(), buf.begin(), buf.end());
        }

        std::string getName() const {
            return _name;
        }

        int64_t getNumGeneralRegisters() const {
            return _numGeneralRegisters;
        }

        const std::vector<intel_npu::vm::Type>& getParamTypes() const {
            return _paramTypes;
        }

        const std::vector<intel_npu::vm::Type>& getResultTypes() const {
            return _resultTypes;
        }

        bool isEntrypoint() const {
            return _isEntrypoint;
        }

        const std::vector<uint8_t>& getBody() const {
            return _body;
        }

    public:
        FunctionBuilder(std::string name, int64_t numGeneralRegisters, std::vector<intel_npu::vm::Type> paramTypes = {},
                        std::vector<intel_npu::vm::Type> resultTypes = {}, bool isEntrypoint = false)
                : _name(std::move(name)),
                  _numGeneralRegisters(numGeneralRegisters),
                  _paramTypes(std::move(paramTypes)),
                  _resultTypes(std::move(resultTypes)),
                  _isEntrypoint(isEntrypoint) {
        }

        // Appends an instruction with the given opcode and 16-bit operands
        FunctionBuilder& instruction(intel_npu::vm::OpCode op, const std::vector<int16_t>& operands) {
            append(static_cast<uint16_t>(op));
            for (auto operand : operands) {
                append(operand);
            }
            return *this;
        }

        // Encode SET_IMM: reg[dst] = imm (64-bit immediate)
        FunctionBuilder& setImm(int16_t dst, int64_t imm) {
            append(static_cast<uint16_t>(intel_npu::vm::OpCode::SET_IMM));
            append(dst);
            append(imm);
            return *this;
        }

        // Encode RETV: return values held in the listed registers
        FunctionBuilder& retv(std::initializer_list<int16_t> resultRegs) {
            append(static_cast<uint16_t>(intel_npu::vm::OpCode::RETV));
            append(static_cast<int16_t>(resultRegs.size()));
            for (auto reg : resultRegs) {
                append(reg);
            }
            return *this;
        }

        // Encode JMP: unconditional jump by the given int64 PC-relative byte offset
        FunctionBuilder& jmp(int64_t offset) {
            append(static_cast<uint16_t>(intel_npu::vm::OpCode::JMP));
            append(offset);
            return *this;
        }

        // Encode JE: jump by the given int64 PC-relative byte offset if lhsReg == rhsReg
        FunctionBuilder& je(int64_t offset, int16_t lhsReg, int16_t rhsReg) {
            append(static_cast<uint16_t>(intel_npu::vm::OpCode::JE));
            append(offset);
            append(lhsReg);
            append(rhsReg);
            return *this;
        }

        // Encode JNE: jump by the given int64 PC-relative byte offset if lhsReg != rhsReg
        FunctionBuilder& jne(int64_t offset, int16_t lhsReg, int16_t rhsReg) {
            append(static_cast<uint16_t>(intel_npu::vm::OpCode::JNE));
            append(offset);
            append(lhsReg);
            append(rhsReg);
            return *this;
        }

        // Encode RET: return with no values
        FunctionBuilder& ret() {
            append(static_cast<uint16_t>(intel_npu::vm::OpCode::RET));
            return *this;
        }

        // Encode CALL: call rs, N, rN..., M, rM...
        FunctionBuilder& call(int16_t functionIndexReg, std::initializer_list<int16_t> dstRegs,
                              std::initializer_list<int16_t> argRegs) {
            append(static_cast<uint16_t>(intel_npu::vm::OpCode::CALL));
            append(functionIndexReg);
            append(static_cast<int16_t>(dstRegs.size()));
            for (auto reg : dstRegs) {
                append(reg);
            }
            append(static_cast<int16_t>(argRegs.size()));
            for (auto reg : argRegs) {
                append(reg);
            }
            return *this;
        }
    };

    // Adds a function to the module using the provided FunctionBuilder, which encodes the function body and metadata
    BytecodeBuilder& addFunction(const FunctionBuilder& functionBuilder) {
        const auto nameIdx = internString(functionBuilder.getName());

        intel_npu::vm::FunctionType functionType;
        functionType.paramTypeIndices.reserve(functionBuilder.getParamTypes().size());
        functionType.resultTypeIndices.reserve(functionBuilder.getResultTypes().size());

        std::vector<intel_npu::vm::FuncParamResType> fnParamTypes;
        fnParamTypes.reserve(functionBuilder.getParamTypes().size());
        for (const auto& paramType : functionBuilder.getParamTypes()) {
            const auto typeIdx = internNonFunctionType(paramType);
            functionType.paramTypeIndices.push_back(typeIdx);
            fnParamTypes.push_back({paramType, static_cast<int64_t>(typeIdx)});
        }

        std::vector<intel_npu::vm::FuncParamResType> fnResultTypes;
        fnResultTypes.reserve(functionBuilder.getResultTypes().size());
        for (const auto& resultType : functionBuilder.getResultTypes()) {
            const auto typeIdx = internNonFunctionType(resultType);
            functionType.resultTypeIndices.push_back(typeIdx);
            fnResultTypes.push_back({resultType, static_cast<int64_t>(typeIdx)});
        }

        const auto funcTypeIdx = internFunctionType(functionType);

        auto body = intel_npu::vm::Span<uint8_t>(const_cast<uint8_t*>(functionBuilder.getBody().data()),
                                                 functionBuilder.getBody().size());
        _functions.push_back(FunctionContext{
                intel_npu::vm::Function(functionBuilder.getName(), functionBuilder.getNumGeneralRegisters(),
                                        functionBuilder.isEntrypoint(), std::move(fnParamTypes),
                                        std::move(fnResultTypes), body, /*copyBody=*/true),
                nameIdx, funcTypeIdx});
        return *this;
    }

    std::optional<uint64_t> getFunctionIndex(std::string_view name) const {
        for (size_t i = 0; i < _functions.size(); ++i) {
            if (_functions[i].definition.getName() == name) {
                return i;
            }
        }
        return std::nullopt;
    }

    // Sets the bytecode version serialized into the output binary
    BytecodeBuilder& setVersion(intel_npu::vm::Version version) {
        _version = version;
        return *this;
    }

    // Builds the complete bytecode module with all added functions.
    // String and type sections are already populated incrementally; only the function section is assembled here.
    std::vector<uint8_t> build() const {
        if (_functions.empty()) {
            return {};
        }

        // Assemble the function section payload from each function's pre-built body
        std::vector<uint8_t> funcSectionPayload;
        intel_npu::vm::details::FunctionSectionInfo funcInfo;
        funcInfo.numFunctions = _functions.size();
        funcInfo.entrypointFunctionIndex = 0;

        // First entrypoint-marked function becomes module entrypoint
        for (size_t idx = 0; idx < _functions.size(); ++idx) {
            if (_functions[idx].definition.isEntrypoint()) {
                funcInfo.entrypointFunctionIndex = idx;
                break;
            }
        }

        for (const auto& ctx : _functions) {
            const auto& body = ctx.definition.getBody();

            intel_npu::vm::details::FunctionSectionInfo::FunctionInfo functionInfo{};
            functionInfo.nameIndex = ctx.nameIndex;
            functionInfo.functionTypeIndex = ctx.functionTypeIndex;
            functionInfo.numGeneralRegisters = ctx.definition.getNumGeneralRegisters();
            functionInfo.bodyOffset = funcSectionPayload.size();
            functionInfo.bodySize = body.size();

            funcSectionPayload.insert(funcSectionPayload.end(), body.begin(), body.end());
            funcInfo.functionInfos.push_back(functionInfo);
        }

        // Build section header table for function, string, and type sections
        intel_npu::vm::SectionHeaderTable sectionHeaderTable;

        intel_npu::vm::SectionHeader funcHeader{};
        funcHeader.type = intel_npu::vm::SectionType::FuncSection;
        funcHeader.nameIndex = 0;
        funcHeader.offset = 0;  // Computed later by `computeOffsets` call
        funcHeader.size = funcSectionPayload.size();
        funcHeader.info = std::make_unique<intel_npu::vm::details::FunctionSectionInfo>(std::move(funcInfo));
        sectionHeaderTable.addSectionHeader(std::move(funcHeader));

        intel_npu::vm::SectionHeader stringHeader{};
        stringHeader.type = intel_npu::vm::SectionType::StringSection;
        stringHeader.nameIndex = 0;
        stringHeader.offset = 0;  // Computed later by `computeOffsets` call
        stringHeader.size = _stringSectionPayload.size();
        stringHeader.info = std::make_unique<intel_npu::vm::details::DataSectionInfo>(_stringSectionInfo);
        sectionHeaderTable.addSectionHeader(std::move(stringHeader));

        intel_npu::vm::SectionHeader typeHeader{};
        typeHeader.type = intel_npu::vm::SectionType::TypeSection;
        typeHeader.nameIndex = 0;
        typeHeader.offset = 0;  // Computed later by `computeOffsets` call
        typeHeader.size = _typeSectionPayload.size();
        typeHeader.info = std::make_unique<intel_npu::vm::details::DataSectionInfo>(_typeSectionInfo);
        sectionHeaderTable.addSectionHeader(std::move(typeHeader));

        // Compute payload offsets using canonical section ordering
        sectionHeaderTable.computeOffsets();

        std::vector<uint8_t> bytecodeBuffer;
        // File layout: magic number, version, section header table, then section payloads
        intel_npu::vm::MagicNumber(intel_npu::vm::MAGIC_NUMBER).appendTo(bytecodeBuffer);
        _version.appendTo(bytecodeBuffer);
        sectionHeaderTable.appendTo(bytecodeBuffer);

        bytecodeBuffer.insert(bytecodeBuffer.end(), funcSectionPayload.begin(), funcSectionPayload.end());
        bytecodeBuffer.insert(bytecodeBuffer.end(), _stringSectionPayload.begin(), _stringSectionPayload.end());
        bytecodeBuffer.insert(bytecodeBuffer.end(), _typeSectionPayload.begin(), _typeSectionPayload.end());

        return bytecodeBuffer;
    }
};

}  // namespace utils
