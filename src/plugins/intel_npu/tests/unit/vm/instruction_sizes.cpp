//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/instructions.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace intel_npu::vm;
namespace {

std::vector<uint8_t> encodeInstruction(OpCode opcode, const std::vector<int16_t>& operands) {
    std::vector<uint8_t> bytes(OPCODE_SIZE + operands.size() * OPERAND_SIZE);
    const auto opcodeValue = static_cast<uint16_t>(opcode);
    std::memcpy(bytes.data(), &opcodeValue, sizeof(opcodeValue));
    std::memcpy(bytes.data() + OPCODE_SIZE, operands.data(), operands.size() * OPERAND_SIZE);
    return bytes;
}

}  // namespace

TEST(VirtualMachineInstructionSizesTest, BufferCreateVariadicSize) {
    // operands: rd, elemType, rank=2, shape0, shape1, stride0, stride1
    const auto rank = static_cast<int16_t>(2);
    const auto bytes = encodeInstruction(OpCode::BUFFER_CREATE, {/*dst=*/1, /*elem_type=*/2, rank, 10, 20, 30, 40});
    EXPECT_EQ(getInstructionSize(OpCode::BUFFER_CREATE, bytes.data(), bytes.size()),
              OPCODE_SIZE + (3 + 2 * rank) * OPERAND_SIZE);
}

TEST(VirtualMachineInstructionSizesTest, BufferSubviewVariadicSize) {
    // operands: rd, src, rank=2, offsets[2], sizes[2], strides[2]
    const auto rank = static_cast<int16_t>(2);
    const auto bytes = encodeInstruction(OpCode::BUFFER_SUBVIEW, {1, 2, rank, 10, 20, 30, 40, 50, 60});
    EXPECT_EQ(getInstructionSize(OpCode::BUFFER_SUBVIEW, bytes.data(), bytes.size()),
              OPCODE_SIZE + (3 + 3 * rank) * OPERAND_SIZE);
}

TEST(VirtualMachineInstructionSizesTest, BufferViewVariadicSize) {
    // operands: dst, src, byte_offset, elem_type, rank=2, shape[2], strides[2]
    const auto rank = static_cast<int16_t>(2);
    const auto bytes = encodeInstruction(
            OpCode::BUFFER_VIEW, {/*dst=*/1, /*src=*/2, /*byte_offset=*/0, /*elem_type=*/3, rank, 10, 20, 30, 40});
    EXPECT_EQ(getInstructionSize(OpCode::BUFFER_VIEW, bytes.data(), bytes.size()),
              OPCODE_SIZE + (5 + 2 * rank) * OPERAND_SIZE);
}

TEST(VirtualMachineInstructionSizesTest, BufferLoadVariadicSize) {
    // operands: dst, src, rank=2, index0, index1
    const auto rank = static_cast<int16_t>(2);
    const auto bytes = encodeInstruction(OpCode::BUFFER_LOAD, {/*dst=*/1, /*src=*/2, rank, 10, 20});
    EXPECT_EQ(getInstructionSize(OpCode::BUFFER_LOAD, bytes.data(), bytes.size()),
              OPCODE_SIZE + (3 + rank) * OPERAND_SIZE);
}

TEST(VirtualMachineInstructionSizesTest, GetInstructionSizeFallsBackToStaticInstructions) {
    const auto bufferGetDimBytes = encodeInstruction(OpCode::BUFFER_GET_DIM, {1, 2, 3});
    EXPECT_EQ(getStaticInstructionSize(OpCode::ADD_I64), OPCODE_SIZE + 3 * OPERAND_SIZE);
    EXPECT_EQ(getInstructionSize(OpCode::BUFFER_GET_DIM, bufferGetDimBytes.data(), bufferGetDimBytes.size()),
              OPCODE_SIZE + 3 * OPERAND_SIZE);
}

TEST(VirtualMachineInstructionSizesTest, VariadicInstructionSizeRejectsMalformedOps) {
    const auto negativeBufferRankBytes =
            encodeInstruction(OpCode::BUFFER_CREATE, {/*dst=*/1, /*elem_type=*/2, /*rank=*/-1});
    EXPECT_FALSE(
            getInstructionSize(OpCode::BUFFER_CREATE, negativeBufferRankBytes.data(), negativeBufferRankBytes.size())
                    .has_value());

    const auto negativeRankBytes = encodeInstruction(OpCode::BUFFER_SUBVIEW, {/*dst=*/1, /*src=*/2, /*rank=*/-1});
    EXPECT_FALSE(
            getInstructionSize(OpCode::BUFFER_SUBVIEW, negativeRankBytes.data(), negativeRankBytes.size()).has_value());

    const auto negativeViewRankBytes = encodeInstruction(
            OpCode::BUFFER_VIEW, {/*dst=*/1, /*src=*/2, /*byte_offset=*/0, /*elem_type=*/3, /*rank=*/-1});
    EXPECT_FALSE(getInstructionSize(OpCode::BUFFER_VIEW, negativeViewRankBytes.data(), negativeViewRankBytes.size())
                         .has_value());

    const auto unknownOpcode = static_cast<OpCode>(0xFFFF);
    EXPECT_FALSE(getInstructionSize(unknownOpcode, negativeRankBytes.data(), negativeRankBytes.size()).has_value());
}

TEST(VirtualMachineInstructionSizesTest, RetvVariadicSize) {
    const auto resultCount = static_cast<int16_t>(2);
    const auto bytes = encodeInstruction(OpCode::RETV, {resultCount, 4, 5});
    EXPECT_EQ(getInstructionSize(OpCode::RETV, bytes.data(), bytes.size()),
              OPCODE_SIZE + (1 + resultCount) * OPERAND_SIZE);
}

TEST(VirtualMachineInstructionSizesTest, CmdListAddKernelVariadicSize) {
    // operands: cmd_list, kernel, N=1, signal0, M=2, wait0, wait1
    const auto bytes = encodeInstruction(OpCode::CMD_LIST_ADD_KERNEL, {1, 2, 1, 10, 2, 11, 12});
    EXPECT_EQ(getInstructionSize(OpCode::CMD_LIST_ADD_KERNEL, bytes.data(), bytes.size()),
              OPCODE_SIZE + 7 * OPERAND_SIZE);
}

TEST(VirtualMachineInstructionSizesTest, CallVariadicSize) {
    const auto bytes = encodeInstruction(OpCode::CALL, {/*rs=*/1, /*N=*/2, /*dst0=*/4, /*dst1=*/5, /*M=*/3,
                                                        /*arg0=*/6, /*arg1=*/7, /*arg2=*/8});
    EXPECT_EQ(getInstructionSize(OpCode::CALL, bytes.data(), bytes.size()), OPCODE_SIZE + 8 * OPERAND_SIZE);
}

TEST(InstructionSizesTest, KernelCreateVariadicSize) {
    const auto n = static_cast<int16_t>(2);
    const auto m = static_cast<int16_t>(1);
    const auto bytes = encodeInstruction(
            OpCode::KERNEL_CREATE, {/*dst=*/0, /*kidx=*/1, /*kNameIdx=*/1, n, /*in0=*/2, /*in1=*/3, m, /*out0=*/4});
    const auto expected = (6 + n + m) * OPERAND_SIZE;
    EXPECT_EQ(getInstructionSize(OpCode::KERNEL_CREATE, bytes.data(), bytes.size()), expected);
}

TEST(InstructionSizesTest, KernelCreateZeroInputsOutputs) {
    const auto bytes =
            encodeInstruction(OpCode::KERNEL_CREATE, {/*dst=*/0, /*kidx=*/1, /*kNameIdx=*/1, /*N=*/0, /*M=*/0});
    const auto expected = 6 * OPERAND_SIZE;  // opcode + dst + kidx + kNameIdx + N + M
    EXPECT_EQ(getInstructionSize(OpCode::KERNEL_CREATE, bytes.data(), bytes.size()), expected);
}

TEST(InstructionSizesTest, KernelCreateRejectsNegativeInputCount) {
    const auto bytes = encodeInstruction(OpCode::KERNEL_CREATE, {/*dst=*/0, /*kidx=*/1, /*kNameIdx=*/1, /*N=*/-1});
    EXPECT_FALSE(getInstructionSize(OpCode::KERNEL_CREATE, bytes.data(), bytes.size()).has_value());
}

TEST(InstructionSizesTest, KernelCreateRejectsNegativeOutputCount) {
    const auto bytes =
            encodeInstruction(OpCode::KERNEL_CREATE, {/*dst=*/0, /*kidx=*/1, /*kNameIdx=*/1, /*N=*/0, /*M=*/-1});
    EXPECT_FALSE(getInstructionSize(OpCode::KERNEL_CREATE, bytes.data(), bytes.size()).has_value());
}

TEST(InstructionSizesTest, KernelCreateRejectsTruncatedBufferBeforeOutputCount) {
    const auto bytes = encodeInstruction(OpCode::KERNEL_CREATE, {/*dst=*/0, /*kidx=*/1, /*kNameIdx=*/1, /*N=*/2});
    EXPECT_FALSE(getInstructionSize(OpCode::KERNEL_CREATE, bytes.data(), bytes.size()).has_value());
}

TEST(VirtualMachineInstructionSizesTest, VariadicInstructionSizeGatesDecodeHeaderInternally) {
    const auto shortBytes = std::vector<uint8_t>(OPCODE_SIZE, 0u);

    EXPECT_FALSE(getInstructionSize(OpCode::RETV, shortBytes.data(), shortBytes.size()).has_value());
    EXPECT_FALSE(getInstructionSize(OpCode::BUFFER_CREATE, shortBytes.data(), shortBytes.size()).has_value());
    EXPECT_FALSE(getInstructionSize(OpCode::BUFFER_LOAD, shortBytes.data(), shortBytes.size()).has_value());
    EXPECT_FALSE(getInstructionSize(OpCode::BUFFER_SUBVIEW, shortBytes.data(), shortBytes.size()).has_value());
    EXPECT_FALSE(getInstructionSize(OpCode::BUFFER_VIEW, shortBytes.data(), shortBytes.size()).has_value());
    EXPECT_FALSE(getInstructionSize(OpCode::KERNEL_CREATE, shortBytes.data(), shortBytes.size()).has_value());
    EXPECT_FALSE(getInstructionSize(OpCode::CMD_LIST_ADD_KERNEL, shortBytes.data(), shortBytes.size()).has_value());
    EXPECT_FALSE(getInstructionSize(OpCode::CALL, shortBytes.data(), shortBytes.size()).has_value());
}
