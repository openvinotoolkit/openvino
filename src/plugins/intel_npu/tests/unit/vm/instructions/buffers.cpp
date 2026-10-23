// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "utils/common.hpp"
#include "vm_runtime/virtual_machine.h"
#include "vm_value_helpers.hpp"

//#include <llvm/ADT/ScopeExit.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace ov::intel_npu::vm;
using namespace utils;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

TEST_F(VirtualMachineInstructionTest, BufferCreateGetDim) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/9,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/6)
                            .setImm(/*dst=*/3, /*imm=*/6)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/1, 2,
                                                                 /*strides=*/3, 4})
                            .setImm(/*dst=*/5, /*imm=*/0)
                            .setImm(/*dst=*/6, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/7, /*buffer=*/0, /*dim=*/5})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/8, /*buffer=*/0, /*dim=*/6})
                            .retv({7, 8});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0), ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, 4);
    EXPECT_EQ(results.at(1).i64, 6);
}

TEST_F(VirtualMachineInstructionTest, BufferSubviewMatchesRequestedSizes) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/16,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/6)
                            .setImm(/*dst=*/3, /*imm=*/6)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/1, 2,
                                                                 /*strides=*/3, 4})
                            .setImm(/*dst=*/5, /*imm=*/1)
                            .setImm(/*dst=*/6, /*imm=*/1)
                            .setImm(/*dst=*/7, /*imm=*/2)
                            .setImm(/*dst=*/8, /*imm=*/3)
                            .setImm(/*dst=*/9, /*imm=*/1)
                            .setImm(/*dst=*/10, /*imm=*/1)
                            .instruction(OpCode::BUFFER_SUBVIEW, {/*dst=*/11, /*src=*/0, /*rank=*/2,
                                                                  /*offsets=*/5, 6,
                                                                  /*sizes=*/7, 8,
                                                                  /*strides=*/9, 10})
                            .setImm(/*dst=*/12, /*imm=*/0)
                            .setImm(/*dst=*/13, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/14, /*buffer=*/11, /*dim=*/12})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/15, /*buffer=*/11, /*dim=*/13})
                            .retv({14, 15});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0), ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, 2);
    EXPECT_EQ(results.at(1).i64, 3);
}

// Smoke test for buffer.store: build a function that allocates an owned i64 memref via buffer.create, stores a
// parameter value at static indices, and returns. Asserts the VM finalizes without halting. No buffer.load opcode
// exists yet, so the stored value cannot be read back; this exercises the executor's metadata lookup, bounds checking,
// permission check, and write path end-to-end through bytecode parsing and execution.
TEST_F(VirtualMachineInstructionTest, BufferStoreExecutes) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kValueParamReg = 7;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/8,
                                                     /*paramTypes=*/{i64},
                                                     /*resultTypes=*/{},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/6)
                            .setImm(/*dst=*/3, /*imm=*/6)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/1, 2,
                                                                 /*strides=*/3, 4})
                            .setImm(/*dst=*/5, /*imm=*/1)
                            .setImm(/*dst=*/6, /*imm=*/2)
                            .instruction(OpCode::BUFFER_STORE, {/*buffer=*/0, /*value=*/kValueParamReg, /*rank=*/2,
                                                                /*indices=*/5, 6})
                            .ret();

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {ov::intel_npu::vm::makeI64(static_cast<int64_t>(0x123456789ABCDEF))};
    std::vector<npu_vm_value> results = {};
    EXPECT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);
}

// Smoke test for buffer.store and buffer.load: build a function that allocates an owned i64 memref via buffer.create,
// stores a parameter value at static indices, reads it back, and returns it. This exercises the executor's metadata
// lookup, bounds checking, permission check, and both read/write paths end-to-end through bytecode parsing and
// execution
TEST_F(VirtualMachineInstructionTest, BufferStoreAndLoadRoundTripsValue) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kValueParamReg = 9;
    constexpr int16_t kLoadedValueReg = 7;
    const auto storedValue = ov::intel_npu::vm::makeI64(static_cast<int64_t>(0x123456789ABCDEF));

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/10,
                                                     /*paramTypes=*/{i64},
                                                     /*resultTypes=*/{i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/1)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .setImm(/*dst=*/3, /*imm=*/1)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .setImm(/*dst=*/5, /*imm=*/0)
                            .setImm(/*dst=*/6, /*imm=*/0)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/1, 2,
                                                                 /*strides=*/3, 4})
                            .instruction(OpCode::BUFFER_STORE, {/*buffer=*/0, /*value=*/kValueParamReg, /*rank=*/2,
                                                                /*indices=*/5, 6})
                            .instruction(OpCode::BUFFER_LOAD, {/*dst=*/kLoadedValueReg, /*buffer=*/0, /*rank=*/2,
                                                               /*indices=*/5, 6})
                            .retv({kLoadedValueReg});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {storedValue};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0)};
    EXPECT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, storedValue.i64);
}

// Build a buffer whose shape is supplied as i64 parameters at call time, then read each dimension back through
// buffer.get_dim. This exercises the runtime path for dynamically-shaped buffers: : the dim values stored in the buffer
// descriptor are whatever shape registers held at buffer.create time, and buffer.get_dim returns those runtime values.
TEST_F(VirtualMachineInstructionTest, BufferCreateGetDimDynamicShape) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kDim0ParamReg = 10;
    constexpr int16_t kDim1ParamReg = 11;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/12,
                                                     /*paramTypes=*/{i64, i64},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/kDim0ParamReg, kDim1ParamReg,
                                                                 /*strides=*/kDim1ParamReg, 2})
                            .setImm(/*dst=*/3, /*imm=*/0)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/5, /*buffer=*/0, /*dim=*/3})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/6, /*buffer=*/0, /*dim=*/4})
                            .retv({5, 6});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {ov::intel_npu::vm::makeI64(7), ov::intel_npu::vm::makeI64(11)};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0), ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, 7);
    EXPECT_EQ(results.at(1).i64, 11);
}

// The source memref's dim 2 and the subview's dim-2 offset are both runtime values; sizes and strides are static.
// This builds a buffer.create with a parameter register in the dim-2 slot followed by a buffer.subview whose dim-2
// offset is the second parameter register. The view's dims must equal the requested sizes [1, 16, 31, 1280] regardless
// of the runtime input values.
TEST_F(VirtualMachineInstructionTest, BufferSubviewDynamicOffsetReturnsRequestedSizes) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kDynSrcDim2ParamReg = 28;
    constexpr int16_t kDynOffsetParamReg = 29;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/30,
                                                     /*paramTypes=*/{i64, i64},
                                                     /*resultTypes=*/{i64, i64, i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/1)
                            .setImm(/*dst=*/2, /*imm=*/16)
                            .setImm(/*dst=*/3, /*imm=*/1280)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/4,
                                                                 /*shape=*/1, 2, kDynSrcDim2ParamReg, 3,
                                                                 /*strides=*/4, 4, 4, 4})
                            .setImm(/*dst=*/6, /*imm=*/0)
                            .setImm(/*dst=*/7, /*imm=*/0)
                            .setImm(/*dst=*/8, /*imm=*/0)
                            .setImm(/*dst=*/9, /*imm=*/1)
                            .setImm(/*dst=*/10, /*imm=*/16)
                            .setImm(/*dst=*/11, /*imm=*/31)
                            .setImm(/*dst=*/12, /*imm=*/1280)
                            .setImm(/*dst=*/13, /*imm=*/1)
                            .setImm(/*dst=*/14, /*imm=*/1)
                            .setImm(/*dst=*/15, /*imm=*/1)
                            .setImm(/*dst=*/16, /*imm=*/1)
                            .instruction(OpCode::BUFFER_SUBVIEW, {/*dst=*/5, /*src=*/0, /*rank=*/4,
                                                                  /*offsets=*/6, 7, kDynOffsetParamReg, 8,
                                                                  /*sizes=*/9, 10, 11, 12,
                                                                  /*strides=*/13, 14, 15, 16})
                            .setImm(/*dst=*/17, /*imm=*/0)
                            .setImm(/*dst=*/18, /*imm=*/1)
                            .setImm(/*dst=*/19, /*imm=*/2)
                            .setImm(/*dst=*/20, /*imm=*/3)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/21, /*buffer=*/5, /*dim=*/17})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/22, /*buffer=*/5, /*dim=*/18})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/23, /*buffer=*/5, /*dim=*/19})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/24, /*buffer=*/5, /*dim=*/20})
                            .retv({21, 22, 23, 24});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {ov::intel_npu::vm::makeI64(100), ov::intel_npu::vm::makeI64(31)};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0), ov::intel_npu::vm::makeI64(0),
                                         ov::intel_npu::vm::makeI64(0), ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, 1);
    EXPECT_EQ(results.at(1).i64, 16);
    EXPECT_EQ(results.at(2).i64, 31);
    EXPECT_EQ(results.at(3).i64, 1280);
}

// Verify that buffer.view creates a view with the exact shape and strides provided.
// Source is a flat 8-element i64 buffer reinterpreted as a 2x4 row-major view.
TEST_F(VirtualMachineInstructionTest, BufferViewMatchesRequestedShape) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/13,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/8)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/1,
                                                                 /*shape=*/1,
                                                                 /*strides=*/2})
                            .setImm(/*dst=*/3, /*imm=*/0)
                            .setImm(/*dst=*/4, /*imm=*/2)
                            .setImm(/*dst=*/5, /*imm=*/4)
                            .setImm(/*dst=*/6, /*imm=*/4)
                            .setImm(/*dst=*/7, /*imm=*/1)
                            .instruction(OpCode::BUFFER_VIEW,
                                         {/*dst=*/8, /*src=*/0, /*byteOffset=*/3, kElemI64TypeIndex, /*rank=*/2,
                                          /*shape=*/4, 5,
                                          /*strides=*/6, 7})
                            .setImm(/*dst=*/9, /*imm=*/0)
                            .setImm(/*dst=*/10, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/11, /*buffer=*/8, /*dim=*/9})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/12, /*buffer=*/8, /*dim=*/10})
                            .retv({11, 12});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0), ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, 2);
    EXPECT_EQ(results.at(1).i64, 4);
}

// Verify buffer.view with runtime-supplied shape registers: both shape dims are function parameters,
// exercising the dynamic-register path through the VM's buffer.view handler.
// Source is a flat 16-element i64 buffer; view shape {rows, cols} = {4, 4} is passed at call time.
TEST_F(VirtualMachineInstructionTest, BufferViewDynamicShapeMatchesRuntimeValues) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int16_t kRowsParamReg = 12;
    constexpr int16_t kColsParamReg = 13;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/14,
                                                     /*paramTypes=*/{i64, i64},
                                                     /*resultTypes=*/{i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/16)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/1,
                                                                 /*shape=*/1,
                                                                 /*strides=*/2})
                            .setImm(/*dst=*/3, /*imm=*/0)
                            .setImm(/*dst=*/4, /*imm=*/1)
                            .instruction(OpCode::BUFFER_VIEW,
                                         {/*dst=*/5, /*src=*/0, /*byteOffset=*/3, kElemI64TypeIndex, /*rank=*/2,
                                          /*shape=*/kRowsParamReg, kColsParamReg,
                                          /*strides=*/kColsParamReg, 4})
                            .setImm(/*dst=*/6, /*imm=*/0)
                            .setImm(/*dst=*/7, /*imm=*/1)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/8, /*buffer=*/5, /*dim=*/6})
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/9, /*buffer=*/5, /*dim=*/7})
                            .retv({8, 9});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {ov::intel_npu::vm::makeI64(4), ov::intel_npu::vm::makeI64(4)};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0), ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, 4);
    EXPECT_EQ(results.at(1).i64, 4);
}

TEST_F(VirtualMachineInstructionTest, BufferViewWithNonZeroByteOffset) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/9,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/4)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/1,
                                                                 /*shape=*/1,
                                                                 /*strides=*/2})
                            // Byte offset 16 skips two i64 elements (2x8 bytes), leaving a 2-element view.
                            .setImm(/*dst=*/3, /*imm=*/16)
                            .setImm(/*dst=*/4, /*imm=*/2)
                            .setImm(/*dst=*/5, /*imm=*/1)
                            .instruction(OpCode::BUFFER_VIEW,
                                         {/*dst=*/6, /*src=*/0, /*byteOffset=*/3, kElemI64TypeIndex, /*rank=*/1,
                                          /*shape=*/4,
                                          /*strides=*/5})
                            .setImm(/*dst=*/7, /*imm=*/0)
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/8, /*buffer=*/6, /*dim=*/7})
                            .retv({8});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    EXPECT_EQ(results.at(0).i64, 2);
}

// Exercise the whole buffer instruction set against a single strided buffer in one function
TEST_F(VirtualMachineInstructionTest, BufferStoreLoadGetDimAndSubviewCompose) {
    const auto i64 = getInt64Type();
    constexpr int16_t kElemI64TypeIndex = 0;

    constexpr int16_t kBufReg = 0;
    constexpr int16_t kViewReg = 1;
    constexpr int16_t kDim0Reg = 2;
    constexpr int16_t kDim1Reg = 3;
    constexpr int16_t kStride0Reg = 4;
    constexpr int16_t kStride1Reg = 5;
    constexpr int16_t kOffset0Reg = 6;
    constexpr int16_t kOffset1Reg = 7;
    constexpr int16_t kSrcDimReg = 8;
    constexpr int16_t kViewDimReg = 9;
    constexpr int16_t kLoadedReg = 10;
    constexpr int16_t kStoredReg = 11;
    constexpr int16_t kLoadIdx0Reg = 12;
    constexpr int16_t kLoadIdx1Reg = 13;

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/14,
                                                     /*paramTypes=*/{},
                                                     /*resultTypes=*/{i64, i64, i64},
                                                     /*isEntrypoint=*/true)
                            .setImm(kDim0Reg, /*imm=*/2)
                            .setImm(kDim1Reg, /*imm=*/3)
                            .setImm(kStride0Reg, /*imm=*/3)
                            .setImm(kStride1Reg, /*imm=*/1)
                            .setImm(kOffset0Reg, /*imm=*/1)
                            .setImm(kOffset1Reg, /*imm=*/1)
                            .setImm(kStoredReg, /*imm=*/42)
                            .setImm(kLoadIdx0Reg, /*imm=*/1)
                            .setImm(kLoadIdx1Reg, /*imm=*/2)
                            // buf = int64_t[2][3], row-major strides (3, 1)
                            .instruction(OpCode::BUFFER_CREATE, {kBufReg, kElemI64TypeIndex, /*rank=*/2,
                                                                 /*shape=*/kDim0Reg, kDim1Reg,
                                                                 /*strides=*/kStride0Reg, kStride1Reg})
                            // buf[1][2] = 42
                            .instruction(OpCode::BUFFER_STORE, {kBufReg, /*value=*/kStoredReg, /*rank=*/2,
                                                                /*indices=*/kLoadIdx0Reg, kLoadIdx1Reg})
                            // loaded = buf[1][2] -> 42
                            .instruction(OpCode::BUFFER_LOAD, {/*dst=*/kLoadedReg, kBufReg, /*rank=*/2,
                                                               /*indices=*/kLoadIdx0Reg, kLoadIdx1Reg})
                            // srcDim = buf.shape[1] -> 3
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/kSrcDimReg, kBufReg, /*dim=*/kOffset1Reg})
                            // view = buf[1:2, 1:3] -> shape (1, 2)
                            .instruction(OpCode::BUFFER_SUBVIEW, {/*dst=*/kViewReg, /*src=*/kBufReg, /*rank=*/2,
                                                                  /*offsets=*/kOffset0Reg, kOffset1Reg,
                                                                  /*sizes=*/kOffset1Reg, kDim0Reg,
                                                                  /*strides=*/kOffset1Reg, kOffset1Reg})
                            // viewDim = view.shape[1] -> 2
                            .instruction(OpCode::BUFFER_GET_DIM, {/*dst=*/kViewDimReg, kViewReg, /*dim=*/kOffset1Reg})
                            .retv({kSrcDimReg, kViewDimReg, kLoadedReg});

    const auto bytecode = BytecodeBuilder{}.addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {};
    auto results = std::vector<npu_vm_value>(3, ov::intel_npu::vm::makeI64(0));
    ASSERT_EQ(callWithResults(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME, args, results), NPU_VM_SUCCESS);

    // Source dim 1 of the 2x3 buffer
    EXPECT_EQ(results.at(0).i64, 3);
    // Subview dim 1 comes from the requested size, not the source
    EXPECT_EQ(results.at(1).i64, 2);
    // Value round-tripped through buffer.store / buffer.load
    EXPECT_EQ(results.at(2).i64, 42);
}

// buffer.copy accepts a copy only when the two buffer descriptors agree, element type included. A memref
// parameter's descriptor is built by extractMemRef rather than by an opcode, so it has to name its element type
// the same way buffer.create / buffer.view / buffer.from_const do. addFunction interns a function's parameter
// types, then its result types, then its function type, in the order functions are added, so the leading helper
// pins i64 to index 0 and leaves the entrypoint's BufferType at index 2. Naming the wrong one therefore fails
// this test instead of passing by coincidence.
TEST_F(VirtualMachineInstructionTest, BufferCopyIntoMemRefParameterMatchesBytecodeBuiltDescriptor) {
    const auto i64 = getInt64Type();

    constexpr int16_t kElemI64TypeIndex = 0;
    constexpr int64_t kNumElements = 4;
    constexpr int64_t kStoredValue = 0x123456789ABCDEF;
    constexpr int16_t kOutputParamReg = 7;

    ov::intel_npu::vm::BufferType bufferType{};
    bufferType.dataTypeIndex = kElemI64TypeIndex;
    bufferType.rank = 1;
    bufferType.shape = {kNumElements};
    bufferType.strides = {1};

    auto elemTypeHolder = BytecodeBuilder::FunctionBuilder("elem_type_holder", /*numGeneralRegisters=*/1,
                                                           /*paramTypes=*/{i64},
                                                           /*resultTypes=*/{},
                                                           /*isEntrypoint=*/false)
                                  .ret();

    auto function = BytecodeBuilder::FunctionBuilder(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                                                     /*numGeneralRegisters=*/8,
                                                     /*paramTypes=*/{ov::intel_npu::vm::Type{bufferType}},
                                                     /*resultTypes=*/{},
                                                     /*isEntrypoint=*/true)
                            .setImm(/*dst=*/1, /*imm=*/kNumElements)
                            .setImm(/*dst=*/2, /*imm=*/1)
                            .instruction(OpCode::BUFFER_CREATE, {/*dst=*/0, kElemI64TypeIndex, /*rank=*/1,
                                                                 /*shape=*/1,
                                                                 /*strides=*/2})
                            .setImm(/*dst=*/3, /*imm=*/kStoredValue)
                            .setImm(/*dst=*/4, /*imm=*/0)
                            .instruction(OpCode::BUFFER_STORE, {/*buffer=*/0, /*value=*/3, /*rank=*/1, /*indices=*/4})
                            .instruction(OpCode::BUFFER_COPY, {/*dst=*/kOutputParamReg, /*src=*/0})
                            .ret();

    const auto bytecode = BytecodeBuilder{}.addFunction(elemTypeHolder).addFunction(function).build();
    ASSERT_EQ(loadEntrypoint(bytecode, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME), NPU_VM_SUCCESS);

    std::vector<int64_t> output(kNumElements, 0);
    std::vector<int64_t> sizes = {kNumElements};
    std::vector<int64_t> strides = {1};

    // Destroy on scope exit so a failing assertion below does not leak the memref
    npu_vm_runtime_mem_ref_handle_t outputMemRef = nullptr;
    ASSERT_EQ(npuVMRuntimeCreateMemRef(/*dimsCount=*/1, &outputMemRef), NPU_VM_RUNTIME_RESULT_SUCCESS);
    ASSERT_EQ(npuVMRuntimeSetMemRef(outputMemRef, output.data(), output.data(), /*offset=*/0, sizes.data(),
                                    strides.data(), /*dimsCount=*/1),
              NPU_VM_RUNTIME_RESULT_SUCCESS);

    npu_vm_runtime_execute_params_t params{};
    params.pOutputs = &outputMemRef;
    params.numOfOutputs = 1;

    EXPECT_EQ(npu_vm_infer(engine, &params), NPU_VM_SUCCESS);
    EXPECT_EQ(output.at(0), kStoredValue);

    EXPECT_EQ(npuVMRuntimeDestroyMemRef(outputMemRef), NPU_VM_RUNTIME_RESULT_SUCCESS);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
