// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/magic_number.hpp"
#include "npu_bytecode_utils/network_description.hpp"
#include "npu_bytecode_utils/print_utils.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "npu_bytecode_utils/version.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace ov;

namespace {

bool checkInstructionInBounds(intel_npu::vm::Span<uint8_t> functionBody, intel_npu::vm::OpCode opcode) {
    const auto instructionSize = intel_npu::vm::getInstructionSize(opcode, functionBody.begin(), functionBody.size());
    if (!instructionSize.has_value()) {
        NPU_VM_LOG_ERROR(
                "Failed to decode instruction size for opcode {}: unknown opcode or malformed instruction body",
                static_cast<uint16_t>(opcode));
        return false;
    }
    if (functionBody.size() < instructionSize.value()) {
        NPU_VM_LOG_ERROR(
                "Reached end of function body while decoding an instruction. This likely means the function body "
                "is malformed");
        return false;
    }
    return true;
}

void printFunctionBody(intel_npu::vm::Span<uint8_t> body, size_t indentLevel = 0) {
    using namespace intel_npu::vm;

    const auto printInstruction = [&](std::string_view opcode, const std::vector<int64_t>& operands = {}) {
        printIndent(indentLevel);
        std::cout << opcode;
        for (size_t i = 0; i < operands.size(); ++i) {
            std::cout << " " << operands.at(i);
            if (i < operands.size() - 1) {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
    };

    while (body.begin() != nullptr && !body.empty()) {
        if (body.size() < intel_npu::vm::OPCODE_SIZE) {
            NPU_VM_LOG_ERROR(
                    "Reached end of function body while decoding an opcode. This likely means the function body "
                    "is malformed.");
            break;
        }
        const auto opcode = intel_npu::vm::getOpcode(body.begin());
        if (!checkInstructionInBounds(body, opcode)) {
            break;
        }
        switch (opcode) {
        case OpCode::ADD_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("add.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MUL_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("mul.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MIN_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("min.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MAX_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("max.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::CMP_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            const auto flag = getOperand(body.begin(), 3);
            printInstruction("cmp.i64", {dstRegNum, srcReg1Num, srcReg2Num, flag});
            break;
        }
        case OpCode::CMP_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            const auto flag = getOperand(body.begin(), 3);
            printInstruction("cmp.f64", {dstRegNum, srcReg1Num, srcReg2Num, flag});
            break;
        }
        case OpCode::SUB_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("sub.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::DIV_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("div.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::DIV_U64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("div.u64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MIN_U64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("min.u64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::ADD_U64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("add.u64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MAX_U64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("max.u64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MUL_U64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("mul.u64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::REM_U64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("rem.u64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::SUB_U64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("sub.u64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::REM_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("rem.i64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::ABS_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("abs.i64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::AND_64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("and.64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::NOT_64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("not.64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::OR_64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("or.64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::XOR_64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("xor.64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::SLL_64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("sll.64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::SRL_64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("srl.64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::SRA_64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("sra.64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::CMD_LIST_CREATE: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            printInstruction("cmd_list.create", {dstRegNum});
            break;
        }
        case OpCode::CMD_LIST_CLOSE: {
            const auto srcRegNum = getOperand(body.begin(), 0);
            printInstruction("cmd_list.close", {srcRegNum});
            break;
        }
        case OpCode::CMD_LIST_EXEC: {
            const auto srcRegNum = getOperand(body.begin(), 0);
            const auto hostSync = getOperand(body.begin(), 1);
            printInstruction("cmd_list.exec", {srcRegNum, hostSync});
            break;
        }
        case OpCode::CMD_LIST_ADD_KERNEL: {
            // Instruction layout: `cmd_list.add_kernel rs1, rs2, N, rN..., M, rM...`
            //   rs1   - command list handle register
            //   rs2   - kernel handle register
            //   N     - immediate: number of signal-event registers
            //   rN... - N signal-event registers
            //   M     - immediate: number of wait-event registers
            //   rM... - M wait-event registers
            const auto cmdListRegNum = getOperand(body.begin(), 0);
            const auto kernelRegNum = getOperand(body.begin(), 1);
            const auto numSignalEvents = getOperand(body.begin(), 2);
            if (numSignalEvents < 0) {
                NPU_VM_LOG_ERROR("Malformed cmd_list.add_kernel instruction: negative signal-event count");
                return;
            }

            std::vector<int64_t> operands = {cmdListRegNum, kernelRegNum, static_cast<int64_t>(numSignalEvents)};
            for (int16_t i = 0; i < numSignalEvents; ++i) {
                operands.push_back(getOperand(body.begin(), 3 + i));
            }

            const auto numWaitEvents = getOperand(body.begin(), 3 + numSignalEvents);
            if (numWaitEvents < 0) {
                NPU_VM_LOG_ERROR("Malformed cmd_list.add_kernel instruction: negative wait-event count");
                return;
            }

            operands.push_back(static_cast<int64_t>(numWaitEvents));
            for (int16_t i = 0; i < numWaitEvents; ++i) {
                operands.push_back(getOperand(body.begin(), 4 + numSignalEvents + i));
            }

            printInstruction("cmd_list.add_kernel", operands);
            break;
        }
        case OpCode::KERNEL_CREATE: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto kidxImm = getOperand(body.begin(), 1);
            const auto kNameIdxImm = getOperand(body.begin(), 2);
            const auto numInputs = getOperand(body.begin(), 3);
            if (numInputs < 0) {
                NPU_VM_LOG_ERROR("Malformed kernel.create instruction: negative input count {}", numInputs);
                return;
            }
            std::vector<int64_t> operands = {dstRegNum, kidxImm, kNameIdxImm, numInputs};
            constexpr auto inputsOffset = 4;
            for (size_t i = 0; i < static_cast<size_t>(numInputs); ++i) {
                operands.push_back(getOperand(body.begin(), inputsOffset + i));
            }
            const auto outputsCountOffset = inputsOffset + static_cast<size_t>(numInputs);
            const auto numOutputs = getOperand(body.begin(), outputsCountOffset);
            if (numOutputs < 0) {
                NPU_VM_LOG_ERROR("Malformed kernel.create instruction: negative output count {}", numOutputs);
                return;
            }
            operands.push_back(numOutputs);
            const auto outputsOffset = outputsCountOffset + 1;
            for (size_t i = 0; i < static_cast<size_t>(numOutputs); ++i) {
                operands.push_back(getOperand(body.begin(), outputsOffset + i));
            }
            printInstruction("kernel.create", operands);
            break;
        }
        case OpCode::SELECT: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto condRegNum = getOperand(body.begin(), 1);
            const auto trueRegNum = getOperand(body.begin(), 2);
            const auto falseRegNum = getOperand(body.begin(), 3);
            printInstruction("select", {dstRegNum, condRegNum, trueRegNum, falseRegNum});
            break;
        }
        case OpCode::CALL: {
            // Instruction layout: `call rs, N, rN..., M, rM...`
            //   rs    - register holding the callee function index
            //   N     - immediate: number of destination registers that will receive the return values
            //   rN... - N destination register operands (in call order; mapped by `retv` on return)
            //   M     - immediate: number of argument registers passed to the callee
            //   rM... - M argument register operands; loaded into callee parameter registers [G, G+M)
            const auto funcIdxReg = getOperand(body.begin(), 0);
            const auto n = getOperand(body.begin(), 1);
            if (n < 0) {
                NPU_VM_LOG_ERROR("Malformed call instruction: negative destination register count");
                return;
            }

            std::vector<int64_t> operands = {funcIdxReg, static_cast<int64_t>(n)};
            // Collect N destination registers (indices 2..2+N-1)
            for (int16_t i = 0; i < n; ++i) {
                operands.push_back(getOperand(body.begin(), 2 + i));
            }
            // M is stored immediately after the N destination registers (index 2+N)
            const auto m = getOperand(body.begin(), 2 + n);
            if (m < 0) {
                NPU_VM_LOG_ERROR("Malformed call instruction: negative argument register count");
                return;
            }

            operands.push_back(static_cast<int64_t>(m));
            // Collect M argument registers (indices 3+N..3+N+M-1)
            for (int16_t i = 0; i < m; ++i) {
                operands.push_back(getOperand(body.begin(), 3 + n + i));
            }
            printInstruction("call", operands);
            break;
        }
        case OpCode::RET: {
            printInstruction("ret");
            break;
        }
        case OpCode::RETV: {
            const auto numOperandsRegNum = getOperand(body.begin(), 0);
            constexpr auto variadicOperandsOffset = 1;
            std::vector<int64_t> operands = {numOperandsRegNum};
            for (size_t i = 0; i < static_cast<size_t>(numOperandsRegNum); ++i) {
                operands.push_back(getOperand(body.begin(), variadicOperandsOffset + i));
            }
            printInstruction("retv", operands);
            break;
        }
        case OpCode::SET: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("set", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::SET_IMM: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto immValue = get64BitImm(body.begin(), 1);
            printInstruction("set.imm", {dstRegNum, immValue});
            break;
        }
        case OpCode::ASSERT: {
            const auto conditionRegNum = getOperand(body.begin(), 0);
            const auto msgSymIndex = getOperand(body.begin(), 1);
            printInstruction("assert", {conditionRegNum, msgSymIndex});
            break;
        }
        case OpCode::BUFFER_CREATE: {
            std::vector<int64_t> operands;
            operands.push_back(getOperand(body.begin(), 0));  // dst
            operands.push_back(getOperand(body.begin(), 1));  // elem_type
            const auto rank = getOperand(body.begin(), 2);
            operands.push_back(rank);
            for (int64_t i = 0; i < 2 * rank; ++i) {
                operands.push_back(getOperand(body.begin(), 3 + i));
            }
            printInstruction("buffer.create", operands);
            break;
        }
        case OpCode::BUFFER_FROM_CONST: {
            std::vector<int64_t> operands;
            operands.push_back(getOperand(body.begin(), 0));  // dst
            operands.push_back(getOperand(body.begin(), 1));  // constant index
            operands.push_back(getOperand(body.begin(), 2));  // elem type
            const auto rank = getOperand(body.begin(), 3);
            operands.push_back(rank);
            for (int64_t i = 0; i < 2 * rank; ++i) {
                operands.push_back(getOperand(body.begin(), 4 + i));
            }
            printInstruction("buffer.from_const", operands);
            break;
        }
        case OpCode::BUFFER_GET_DIM: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto bufRegNum = getOperand(body.begin(), 1);
            const auto dimRegNum = getOperand(body.begin(), 2);
            printInstruction("buffer.get_dim", {dstRegNum, bufRegNum, dimRegNum});
            break;
        }
        case OpCode::BUFFER_LOAD: {
            std::vector<int64_t> operands;
            operands.push_back(getOperand(body.begin(), 0));  // dst
            operands.push_back(getOperand(body.begin(), 1));  // buffer
            const auto rank = getOperand(body.begin(), 2);
            operands.push_back(rank);
            for (int64_t i = 0; i < rank; ++i) {
                operands.push_back(getOperand(body.begin(), 3 + i));
            }
            printInstruction("buffer.load", operands);
            break;
        }
        case OpCode::BUFFER_SUBVIEW: {
            std::vector<int64_t> operands;
            operands.push_back(getOperand(body.begin(), 0));  // dst
            operands.push_back(getOperand(body.begin(), 1));  // src
            const auto rank = getOperand(body.begin(), 2);
            operands.push_back(rank);
            constexpr auto offsetsStartIndex = 3;
            for (int64_t i = 0; i < 3 * rank; ++i) {
                operands.push_back(getOperand(body.begin(), offsetsStartIndex + i));
            }
            printInstruction("buffer.subview", operands);
            break;
        }
        case OpCode::BUFFER_VIEW: {
            std::vector<int64_t> operands;
            operands.push_back(getOperand(body.begin(), 0));  // dst
            operands.push_back(getOperand(body.begin(), 1));  // src
            operands.push_back(getOperand(body.begin(), 2));  // byte_offset
            operands.push_back(getOperand(body.begin(), 3));  // elem_type
            const auto rank = getOperand(body.begin(), 4);
            operands.push_back(rank);
            constexpr auto shapeStartIndex = 5;
            for (int64_t i = 0; i < 2 * rank; ++i) {
                operands.push_back(getOperand(body.begin(), shapeStartIndex + i));
            }
            printInstruction("buffer.view", operands);
            break;
        }
        case OpCode::BUFFER_STORE: {
            std::vector<int64_t> operands;
            operands.push_back(getOperand(body.begin(), 0));  // buffer
            operands.push_back(getOperand(body.begin(), 1));  // value
            const auto rank = getOperand(body.begin(), 2);
            operands.push_back(rank);
            for (int64_t i = 0; i < rank; ++i) {
                operands.push_back(getOperand(body.begin(), 3 + i));
            }
            printInstruction("buffer.store", operands);
            break;
        }
        case OpCode::BUFFER_COPY: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("buffer.copy", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::ADD_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("add.f64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::SUB_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("sub.f64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MUL_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("mul.f64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::DIV_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("div.f64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::REM_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("rem.f64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MAX_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("max.f64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::MIN_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcReg1Num = getOperand(body.begin(), 1);
            const auto srcReg2Num = getOperand(body.begin(), 2);
            printInstruction("min.f64", {dstRegNum, srcReg1Num, srcReg2Num});
            break;
        }
        case OpCode::ABS_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("abs.f64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::NEG_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("neg.f64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CEIL_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("ceil.f64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::FLOOR_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("floor.f64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::ROUND_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            const auto immValue = getOperand(body.begin(), 2);
            printInstruction("round.f64", {dstRegNum, srcRegNum, immValue});
            break;
        }
        case OpCode::JMP: {
            const auto offset = get64BitImm(body.begin(), 0);
            printInstruction("jmp", {offset});
            break;
        }
        case OpCode::JE: {
            const auto offset = get64BitImm(body.begin(), 0);
            const auto lhs = getAsOperand<int16_t>(body.begin(), OPCODE_SIZE + sizeof(int64_t));
            const auto rhs = getAsOperand<int16_t>(body.begin(), OPCODE_SIZE + sizeof(int64_t) + sizeof(int16_t));
            printInstruction("je", {offset, lhs, rhs});
            break;
        }
        case OpCode::JNE: {
            const auto offset = get64BitImm(body.begin(), 0);
            const auto lhs = getAsOperand<int16_t>(body.begin(), OPCODE_SIZE + sizeof(int64_t));
            const auto rhs = getAsOperand<int16_t>(body.begin(), OPCODE_SIZE + sizeof(int64_t) + sizeof(int16_t));
            printInstruction("jne", {offset, lhs, rhs});
            break;
        }
        case OpCode::CONVERT_I8_TO_F32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i8tof32", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I8_TO_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i8tof64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I16_TO_I8: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i16toi8", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I16_TO_F32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i16tof32", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I16_TO_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i16tof64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I32_TO_I8: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i32toi8", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I32_TO_I16: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i32toi16", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I32_TO_F32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i32tof32", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I32_TO_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i32tof64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I64_TO_I8: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i64toi8", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I64_TO_I16: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i64toi16", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I64_TO_I32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i64toi32", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I64_TO_F32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i64tof32", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_I64_TO_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.i64tof64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F32_TO_I8: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f32toi8", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F32_TO_I16: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f32toi16", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F32_TO_I32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f32toi32", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F32_TO_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f32toi64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F32_TO_F64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f32tof64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F64_TO_I8: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f64toi8", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F64_TO_I16: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f64toi16", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F64_TO_I32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f64toi32", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F64_TO_I64: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f64toi64", {dstRegNum, srcRegNum});
            break;
        }
        case OpCode::CONVERT_F64_TO_F32: {
            const auto dstRegNum = getOperand(body.begin(), 0);
            const auto srcRegNum = getOperand(body.begin(), 1);
            printInstruction("convert.f64tof32", {dstRegNum, srcRegNum});
            break;
        }
        default:
            NPU_VM_LOG_ERROR("Unknown opcode {}", static_cast<uint16_t>(opcode));
            return;
        }
        const auto instructionSize = getInstructionSize(opcode, body.begin(), body.size());
        if (!instructionSize.has_value()) {
            NPU_VM_LOG_ERROR(
                    "Failed to decode instruction size for opcode {}. This likely means the instruction encoding "
                    "is malformed",
                    static_cast<uint16_t>(opcode));
            return;
        }
        body = body.subspan(instructionSize.value());
    }
}

void printConstant(intel_npu::vm::Span<uint8_t> constant, bool printFull) {
    const auto printEntireConstant = [&]() {
        for (const auto& byte : constant) {
            std::cout << intel_npu::vm::toHex(byte);
        }
    };

    const auto printPartialConstant = [&]() {
        constexpr size_t numBytesToPrintBeforeAndAfter = 4;
        if (constant.size() <= numBytesToPrintBeforeAndAfter * 2) {
            printEntireConstant();
            return;
        }
        for (size_t i = 0; i < numBytesToPrintBeforeAndAfter; ++i) {
            std::cout << intel_npu::vm::toHex(constant.at(i));
        }
        std::cout << "...";
        for (size_t i = constant.size() - numBytesToPrintBeforeAndAfter; i < constant.size(); ++i) {
            std::cout << intel_npu::vm::toHex(constant.at(i));
        }
    };

    if (constant.empty()) {
        std::cout << "(empty)";
        return;
    }
    std::cout << "0x";
    if (printFull) {
        printEntireConstant();
    } else {
        // Only print the first and last few elements when printFull is disabled
        printPartialConstant();
    }
}

void printString(intel_npu::vm::Span<uint8_t> string) {
    if (string.empty()) {
        std::cout << "\"\"";
        return;
    }
    for (const auto& byte : string) {
        if (byte == '\0') {
            // Ensure the null terminator character is printed as well
            std::cout << "\\0";
            continue;
        }
        std::cout << byte;
    }
}

void printType(intel_npu::vm::Span<uint8_t> type) {
    if (type.empty()) {
        std::cout << "(empty)";
        return;
    }
    auto typeCode = static_cast<intel_npu::vm::TypeCode>(type.at(0));
    auto typeData = type.subspan(1);

    switch (typeCode) {
    case intel_npu::vm::TypeCode::INTEGER: {
        intel_npu::vm::IntegerType intType{};
        if (!intType.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse integer type from type data");
            std::cout << "i<invalid>";
            return;
        }
        intType.print();
        break;
    }
    case intel_npu::vm::TypeCode::FLOAT: {
        intel_npu::vm::FloatType floatType{};
        if (!floatType.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse float type from type data");
            std::cout << "f<invalid>";
            return;
        }
        floatType.print();
        break;
    }
    case intel_npu::vm::TypeCode::OPAQUE: {
        intel_npu::vm::OpaqueType opaqueType{};
        if (!opaqueType.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse opaque type from type data");
            std::cout << "opaque<invalid>";
            return;
        }
        opaqueType.print();
        break;
    }
    case intel_npu::vm::TypeCode::BUFFER: {
        intel_npu::vm::BufferType bufferType{};
        if (!bufferType.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse buffer type from type data");
            std::cout << "buffer<invalid>";
            return;
        }
        bufferType.print();
        break;
    }
    case intel_npu::vm::TypeCode::FUNCTION: {
        intel_npu::vm::FunctionType funcType{};
        if (!funcType.parseFrom(typeData)) {
            NPU_VM_LOG_ERROR("Failed to parse function type from type data");
            std::cout << "function<invalid>";
            return;
        }
        funcType.print();
        break;
    }
    default: {
        std::cout << "Unknown type code: " << intel_npu::vm::toHex(static_cast<uint8_t>(typeCode));
        break;
    }
    }
}

}  // namespace

intel_npu::vm::Version intel_npu::vm::BytecodeReader::getMinSupportedVersion() {
    return Version{1, 0, 0};
}

intel_npu::vm::Version intel_npu::vm::BytecodeReader::getMaxSupportedVersion() {
    return Version{1, 2, 0};
}

intel_npu::vm::SectionHeaderTable& intel_npu::vm::BytecodeReader::getSectionHeaderTable() {
    return _sectionHeaderTable;
}

const std::vector<intel_npu::vm::Span<uint8_t>>& intel_npu::vm::BytecodeReader::getSections() const {
    return _sections;
}

intel_npu::vm::Span<uint8_t> intel_npu::vm::BytecodeReader::getDataSectionEntry(intel_npu::vm::SectionType sectionType,
                                                                                size_t entryIndex) const {
    const auto& sectionHeaders = _sectionHeaderTable.getSectionHeaders();
    const auto& sections = getSections();
    for (size_t i = 0; i < sectionHeaders.size(); ++i) {
        const auto& header = sectionHeaders.at(i);
        if (header.type != sectionType) {
            continue;
        }
        auto dataSectionInfo = dynamic_cast<const intel_npu::vm::details::DataSectionInfo*>(header.info.get());
        if (dataSectionInfo == nullptr) {
            NPU_VM_LOG_ERROR("Section header for section of type {} does not contain data section info",
                             static_cast<int>(sectionType));
            return {};
        }
        if (entryIndex >= dataSectionInfo->numData) {
            NPU_VM_LOG_ERROR("Entry index {} out of bounds for section with {} entries", entryIndex,
                             dataSectionInfo->numData);
            return {};
        }
        if (entryIndex >= dataSectionInfo->dataInfos.size()) {
            NPU_VM_LOG_ERROR("Entry index {} out of bounds for section with {} data infos", entryIndex,
                             dataSectionInfo->dataInfos.size());
            return {};
        }
        if (i >= sections.size()) {
            NPU_VM_LOG_ERROR("Could not find associated section with section header {}", i);
            return {};
        }
        const auto& dataInfo = dataSectionInfo->dataInfos.at(entryIndex);
        auto& sectionContent = sections.at(i);
        if (dataInfo.offset > sectionContent.size() || dataInfo.size > sectionContent.size() - dataInfo.offset) {
            NPU_VM_LOG_ERROR("Entry with offset {} and size {} exceeds section bounds", dataInfo.offset, dataInfo.size);
            return {};
        }
        return sectionContent.subspan(dataInfo.offset, dataInfo.size);
    }
    return {};
}

std::optional<std::string> intel_npu::vm::BytecodeReader::getString(size_t stringIndex) const {
    auto stringEntry = getDataSectionEntry(SectionType::StringSection, stringIndex);
    if (stringEntry.empty()) {
        return std::nullopt;
    }
    auto stringSize = stringEntry.size();
    // Remove the null terminator from the end of the string, since std::string does not require a null terminator
    if (stringSize > 0 && stringEntry.at(stringSize - 1) == '\0') {
        --stringSize;
    }
    auto subString = stringEntry.subspan(0, stringSize);
    return std::string(subString.begin(), subString.end());
}

std::optional<intel_npu::vm::FunctionType> intel_npu::vm::BytecodeReader::getFunctionType(size_t typeIndex) const {
    auto typeEntry = getDataSectionEntry(SectionType::TypeSection, typeIndex);
    if (typeEntry.empty()) {
        return std::nullopt;
    }
    intel_npu::vm::FunctionType funcType{};
    const auto typeCode = static_cast<TypeCode>(typeEntry.at(0));
    if (typeCode != TypeCode::FUNCTION) {
        NPU_VM_LOG_ERROR("Type entry at index {} has invalid type code {}, expected FUNCTION type code", typeIndex,
                         static_cast<uint8_t>(typeCode));
        return std::nullopt;
    }
    typeEntry = typeEntry.subspan(sizeof(intel_npu::vm::TypeCode));
    if (!funcType.parseFrom(typeEntry)) {
        NPU_VM_LOG_ERROR("Failed to parse function type at index {}", typeIndex);
        return std::nullopt;
    }
    return funcType;
}

bool intel_npu::vm::BytecodeReader::isVersionSupported(intel_npu::vm::Span<uint8_t> bytecode, Version minVersion,
                                                       Version maxVersion) {
    MagicNumber magicNumber{};
    if (!magicNumber.parseFrom(bytecode)) {
        NPU_VM_LOG_ERROR("Failed to parse magic number");
        return false;
    }
    Version version{};
    if (!version.parseFrom(bytecode)) {
        NPU_VM_LOG_ERROR("Failed to parse version");
        return false;
    }

    // Patch version is ignored for compatibility checking
    maxVersion = maxVersion.resetPatchNumber();
    minVersion = minVersion.resetPatchNumber();
    const auto bytecodeVersion = version.resetPatchNumber();

    // Bytecode version must not exceed the VM's current version
    // A higher version means the bytecode uses features the VM does not yet support
    if (bytecodeVersion > maxVersion) {
        NPU_VM_LOG_ERROR(
                "Bytecode version {} exceeds VM version {}. The bytecode requires a newer VM than the one available",
                version, maxVersion);
        return false;
    }

    // Bytecode version must not be below the minimum supported version
    // The runtime maintains backward compatibility, but deprecated formats may be dropped
    if (bytecodeVersion < minVersion) {
        NPU_VM_LOG_ERROR("Bytecode version {} is below the minimum supported version {}. This bytecode format has been "
                         "deprecated",
                         version, minVersion);
        return false;
    }

    return true;
}

bool intel_npu::vm::BytecodeReader::isVersionSupported() const {
    return isVersionSupported(_bytecode);
}

bool intel_npu::vm::BytecodeReader::parseFileHeader() {
    // Wrap the raw bytecode in a Span that tracks the current read position.
    // Each parseFrom() call advances the span past the consumed bytes.
    auto bytecodeBuffer = _bytecode;

    if (!_magicNumber.parseFrom(bytecodeBuffer)) {
        NPU_VM_LOG_ERROR("Failed to parse magic number");
        return false;
    }

    if (!_version.parseFrom(bytecodeBuffer)) {
        NPU_VM_LOG_ERROR("Failed to parse version");
        return false;
    }

    if (!_sectionHeaderTable.parseFrom(bytecodeBuffer)) {
        NPU_VM_LOG_ERROR("Failed to parse section header table");
        return false;
    }

    return true;
}

bool intel_npu::vm::BytecodeReader::parseSections() {
    auto bytecodeBuffer = _bytecode;

    for (const auto& header : _sectionHeaderTable.getSectionHeaders()) {
        const auto sectionType = header.type;
        const auto offset = header.offset;
        const auto size = header.size;
        if (offset > bytecodeBuffer.size() || size > bytecodeBuffer.size() - offset) {
            NPU_VM_LOG_ERROR(
                    "Section of type {} has invalid offset {} and size {} that exceed bytecode buffer (size {})",
                    static_cast<uint64_t>(sectionType), offset, size, bytecodeBuffer.size());
            return false;
        }
        _sections.push_back(bytecodeBuffer.subspan(offset, size));
    }

    return true;
}

void intel_npu::vm::BytecodeReader::printFunctionSection(const intel_npu::vm::SectionHeader& sectionHeader,
                                                         intel_npu::vm::Span<uint8_t> sectionContent, size_t sectionIdx,
                                                         size_t indentLevel) {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Function section " << sectionIdx << std::endl;
    const auto funcSectionInfo =
            dynamic_cast<const intel_npu::vm::details::FunctionSectionInfo*>(sectionHeader.info.get());
    if (funcSectionInfo == nullptr) {
        NPU_VM_LOG_ERROR("Function section header has invalid info type");
        return;
    }
    for (const auto& funcInfo : funcSectionInfo->functionInfos) {
        intel_npu::vm::printIndent(indentLevel + 1);
        auto functionName = getString(funcInfo.nameIndex);
        if (!functionName.has_value()) {
            NPU_VM_LOG_ERROR("Failed to retrieve function name string for function with name index {}",
                             funcInfo.nameIndex);
        }
        std::cout << "Function name: " << functionName.value_or("<unknown>") << std::endl;
        if (funcInfo.bodyOffset > sectionContent.size() ||
            funcInfo.bodySize > sectionContent.size() - funcInfo.bodyOffset) {
            NPU_VM_LOG_ERROR("Function {} has invalid body offset {} and size {} that exceed section buffer (size {})",
                             functionName.value_or("<unknown>"), funcInfo.bodyOffset, funcInfo.bodySize,
                             sectionContent.size());
            continue;
        }
        const auto body = sectionContent.subspan(funcInfo.bodyOffset, funcInfo.bodySize);
        printFunctionBody(body, indentLevel + 2);
    }
}

void intel_npu::vm::BytecodeReader::printDataSection(const intel_npu::vm::SectionHeader& sectionHeader,
                                                     intel_npu::vm::Span<uint8_t> sectionContent, size_t sectionIdx,
                                                     intel_npu::vm::SectionType sectionType, bool printFull,
                                                     size_t indentLevel) {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << intel_npu::vm::getSectionTypeString(sectionType) << " section " << sectionIdx << std::endl;

    const auto dataSectionInfo = dynamic_cast<intel_npu::vm::details::DataSectionInfo*>(sectionHeader.info.get());
    if (dataSectionInfo == nullptr) {
        NPU_VM_LOG_ERROR("Section header has invalid info type");
        return;
    }
    for (size_t cstIdx = 0; cstIdx < dataSectionInfo->dataInfos.size(); ++cstIdx) {
        const auto& dataInfo = dataSectionInfo->dataInfos.at(cstIdx);
        intel_npu::vm::printIndent(indentLevel + 1);
        std::cout << intel_npu::vm::getSectionTypeString(sectionType) << " " << cstIdx << ": ";
        if (dataInfo.offset > sectionContent.size() || dataInfo.size > sectionContent.size() - dataInfo.offset) {
            NPU_VM_LOG_ERROR("Entry with offset {} and size {} exceeds section bounds", dataInfo.offset, dataInfo.size);
            continue;
        }
        const auto content = sectionContent.subspan(dataInfo.offset, dataInfo.size);
        if (sectionType == intel_npu::vm::SectionType::ConstantSection ||
            sectionType == intel_npu::vm::SectionType::KernelSection) {
            printConstant(content, printFull);
        } else if (sectionType == intel_npu::vm::SectionType::StringSection) {
            printString(content);
        } else if (sectionType == intel_npu::vm::SectionType::TypeSection) {
            printType(content);
        } else if (sectionType == intel_npu::vm::SectionType::MetadataSection) {
            intel_npu::vm::printMetadataEntry(content);
        } else {
            NPU_VM_LOG_ERROR("Unsupported section type for body section printing: {}",
                             static_cast<uint64_t>(sectionType));
        }
        std::cout << std::endl;
    }
}

bool intel_npu::vm::BytecodeReader::parseFile() {
    if (!isVersionSupported()) {
        return false;
    }
    if (!parseFileHeader()) {
        NPU_VM_LOG_ERROR("Failed to parse file header");
        return false;
    }
    if (!parseSections()) {
        NPU_VM_LOG_ERROR("Failed to parse sections");
        return false;
    }
    return true;
}

void intel_npu::vm::BytecodeReader::printFileHeader(size_t indentLevel) {
    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Magic Number: ";
    _magicNumber.print();
    std::cout << std::endl;

    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Version: ";
    _version.print();
    std::cout << std::endl;

    intel_npu::vm::printIndent(indentLevel);
    std::cout << "Section Header Table:" << std::endl;
    _sectionHeaderTable.print(indentLevel + 1);
}

void intel_npu::vm::BytecodeReader::printFile(bool printFull, size_t indentLevel) {
    printFileHeader(indentLevel);

    size_t functionSectionIdx = 0;
    size_t constantSectionIdx = 0;
    size_t kernelSectionIdx = 0;
    size_t stringSectionIdx = 0;
    size_t typeSectionIdx = 0;
    size_t metadataSectionIdx = 0;

    auto& sectionHeaders = _sectionHeaderTable.getSectionHeaders();
    for (size_t i = 0; i < sectionHeaders.size(); ++i) {
        if (i >= _sections.size()) {
            NPU_VM_LOG_ERROR("Could not find associated section with section header {}", i);
            continue;
        }
        auto sectionContent = _sections.at(i);
        const auto& header = sectionHeaders.at(i);
        if (header.type == SectionType::FuncSection) {
            printFunctionSection(header, sectionContent, functionSectionIdx++, indentLevel + 1);
        } else if (header.type == SectionType::ConstantSection) {
            printDataSection(header, sectionContent, constantSectionIdx++, header.type, printFull, indentLevel + 1);
        } else if (header.type == SectionType::StringSection) {
            printDataSection(header, sectionContent, stringSectionIdx++, header.type, printFull, indentLevel + 1);
        } else if (header.type == SectionType::KernelSection) {
            printDataSection(header, sectionContent, kernelSectionIdx++, header.type, printFull, indentLevel + 1);
        } else if (header.type == SectionType::TypeSection) {
            printDataSection(header, sectionContent, typeSectionIdx++, header.type, printFull, indentLevel + 1);
        } else if (header.type == SectionType::MetadataSection) {
            printDataSection(header, sectionContent, metadataSectionIdx++, header.type, printFull, indentLevel + 1);
        } else {
            NPU_VM_LOG_ERROR("Unsupported section type for printing: {}", static_cast<uint64_t>(header.type));
        }
    }
}
