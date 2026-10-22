//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vm_runtime/virtual_machine.h"
#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vm_value_helpers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

using namespace ov::intel_npu::vm;
using namespace utils;

constexpr auto ENTRY_FUNC_NAME = "main";
constexpr auto ADD_FUNC_NAME = "add_i64";
constexpr auto UINT32_IDENTITY_FUNC_NAME = "identity_u32";
constexpr auto BUFFER_IDENTITY_FUNC_NAME = "buffer_identity";
constexpr auto MISSING_FUNC_NAME = "missing";

namespace {

class VirtualMachineApiTest : public testing::Test {
protected:
    npu_vm_module* module{nullptr};
    npu_vm_engine* engine{nullptr};

    void TearDown() override {
        if (engine != nullptr) {
            npu_vm_destroy_engine(engine);
        }
        if (module != nullptr) {
            npu_vm_destroy_module(module);
        }
    }

    std::vector<uint8_t> makeValidBytecode() {
        const auto int64Type = Type{IntegerType{sizeof(int64_t) * CHAR_BIT, /*isSigned=*/true}};
        const auto uint32Type = Type{IntegerType{sizeof(uint32_t) * CHAR_BIT, /*isSigned=*/false}};
        const auto bufferType = Type{BufferType{/*dataTypeIndex=*/0, /*rank=*/1, /*shape=*/{1}, /*strides=*/{1}}};
        return BytecodeBuilder{}
                .addFunction(BytecodeBuilder::FunctionBuilder(ENTRY_FUNC_NAME, /*numGeneralRegisters=*/1,
                                                              /*paramTypes=*/{},
                                                              /*resultTypes=*/{},
                                                              /*isEntrypoint=*/true)
                                     .ret())
                .addFunction(BytecodeBuilder::FunctionBuilder(ADD_FUNC_NAME, /*numGeneralRegisters=*/3,
                                                              /*paramTypes=*/{int64Type, int64Type},
                                                              /*resultTypes=*/{int64Type})
                                     .instruction(OpCode::ADD_I64, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                     .retv({0}))
                .addFunction(BytecodeBuilder::FunctionBuilder(UINT32_IDENTITY_FUNC_NAME, /*numGeneralRegisters=*/1,
                                                              /*paramTypes=*/{uint32Type},
                                                              /*resultTypes=*/{uint32Type})
                                     .retv({0}))
                .addFunction(BytecodeBuilder::FunctionBuilder(BUFFER_IDENTITY_FUNC_NAME, /*numGeneralRegisters=*/1,
                                                              /*paramTypes=*/{bufferType},
                                                              /*resultTypes=*/{bufferType})
                                     .retv({0}))
                .build();
    }
};

TEST_F(VirtualMachineApiTest, ParseReturnsInvalidNullPointerForNullModulePointer) {
    const auto bytecode = makeValidBytecode();
    EXPECT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, ParseSucceedsForValidBytecode) {
    const auto bytecode = makeValidBytecode();
    EXPECT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, ParseFailsForInvalidBytecode) {
    const std::array<uint8_t, 4> badBytecode{0x01, 0x02, 0x03, 0x04};
    EXPECT_EQ(npu_vm_parse_module(badBytecode.data(), static_cast<uint32_t>(badBytecode.size()), &module), NPU_VM_ERROR_UNKNOWN);
}

TEST_F(VirtualMachineApiTest, DestroyModuleReturnsInvalidNullPointerForNullModule) {
    EXPECT_EQ(npu_vm_destroy_module(nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, DestroyModuleSucceedsForValidModule) {
    const auto bytecode = makeValidBytecode();
    npu_vm_module* localModule{nullptr};
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &localModule), NPU_VM_SUCCESS);
    ASSERT_NE(localModule, nullptr);
    EXPECT_EQ(npu_vm_destroy_module(localModule), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, PrintSucceedsForValidBytecode) {
    const auto bytecode = makeValidBytecode();
    EXPECT_EQ(npu_vm_print(bytecode.data(), static_cast<uint32_t>(bytecode.size()), /*print_full=*/false, /*indent_level=*/0), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, PrintFailsForInvalidBytecode) {
    const std::array<uint8_t, 2> badBytecode{0x0A, 0x0B};
    EXPECT_EQ(npu_vm_print(badBytecode.data(), static_cast<uint32_t>(badBytecode.size()), /*print_full=*/false, /*indent_level=*/0),
              NPU_VM_ERROR_UNKNOWN);
}

TEST_F(VirtualMachineApiTest, GetFunctionInfoReturnsInvalidNullPointerForNullInputs) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);

    npu_vm_function_info* info{nullptr};
    EXPECT_EQ(npu_vm_get_function_info(nullptr, ADD_FUNC_NAME, &info), NPU_VM_ERROR_INVALID_NULL_POINTER);
    EXPECT_EQ(npu_vm_get_function_info(module, nullptr, &info), NPU_VM_ERROR_INVALID_NULL_POINTER);
    EXPECT_EQ(npu_vm_get_function_info(module, ADD_FUNC_NAME, nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, GetFunctionInfoReturnsNotFoundForMissingFunction) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);

    npu_vm_function_info* info{nullptr};
    EXPECT_EQ(npu_vm_get_function_info(module, MISSING_FUNC_NAME, &info), NPU_VM_ERROR_FUNCTION_NOT_FOUND);
}

TEST_F(VirtualMachineApiTest, GetFunctionInfoReturnsFunctionSignature) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);

    npu_vm_function_info* info{nullptr};
    ASSERT_EQ(npu_vm_get_function_info(module, ADD_FUNC_NAME, &info), NPU_VM_SUCCESS);
    ASSERT_NE(info, nullptr);
    const std::string expectedName(ADD_FUNC_NAME);
    EXPECT_STREQ(info->name, expectedName.c_str());
    EXPECT_EQ(info->num_params, 2u);
    EXPECT_EQ(info->num_results, 1u);
    EXPECT_EQ(info->param_types[0], npu_vm_type_int64);
    EXPECT_EQ(info->param_types[1], npu_vm_type_int64);
    EXPECT_EQ(info->result_types[0], npu_vm_type_int64);
    EXPECT_EQ(npu_vm_destroy_function_info(info), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, GetFunctionInfoReturnsBufferSignature) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);

    npu_vm_function_info* info{nullptr};
    ASSERT_EQ(npu_vm_get_function_info(module, BUFFER_IDENTITY_FUNC_NAME, &info), NPU_VM_SUCCESS);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->num_params, 1u);
    EXPECT_EQ(info->num_results, 1u);
    EXPECT_EQ(info->param_types[0], npu_vm_type_buffer);
    EXPECT_EQ(info->result_types[0], npu_vm_type_buffer);
    EXPECT_EQ(npu_vm_destroy_function_info(info), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, GetFunctionInfoReturnsUnsignedIntegerSignature) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);

    npu_vm_function_info* info{nullptr};
    ASSERT_EQ(npu_vm_get_function_info(module, UINT32_IDENTITY_FUNC_NAME, &info), NPU_VM_SUCCESS);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->num_params, 1u);
    EXPECT_EQ(info->num_results, 1u);
    EXPECT_EQ(info->param_types[0], npu_vm_type_uint32);
    EXPECT_EQ(info->result_types[0], npu_vm_type_uint32);
    EXPECT_EQ(npu_vm_destroy_function_info(info), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, DestroyFunctionInfoReturnsInvalidNullPointerForNullInfo) {
    EXPECT_EQ(npu_vm_destroy_function_info(nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, CreateNewEngineReturnsInvalidNullPointerForNullEngineHandle) {
    EXPECT_EQ(npu_vm_new_engine(nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, CreateNewEngineSucceeds) {
    EXPECT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    EXPECT_NE(engine, nullptr);
}

TEST_F(VirtualMachineApiTest, DestroyEngineReturnsInvalidNullPointerForNullEngine) {
    EXPECT_EQ(npu_vm_destroy_engine(nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, LoadModuleReturnsInvalidNullPointerForNullEngineOrModule) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);

    EXPECT_EQ(npu_vm_load_module(nullptr, /*moduleParams*/ nullptr, module), NPU_VM_ERROR_INVALID_NULL_POINTER);
    EXPECT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, LoadModuleSucceedsForValidEngineAndModule) {
    const auto bytecode = makeValidBytecode();
    npu_vm_module* localModule{nullptr};
    npu_vm_engine* localEngine{nullptr};

    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &localModule), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&localEngine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(localEngine, /*moduleParams*/ nullptr, localModule), NPU_VM_SUCCESS);

    EXPECT_EQ(npu_vm_destroy_engine(localEngine), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_destroy_module(localModule), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, ResetStateReturnsInvalidNullPointerForNullEngine) {
    EXPECT_EQ(npu_vm_reset_state(nullptr, /*resetExecutionContext=*/false), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, ResetStateSucceedsForValidEngine) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_reset_state(engine, /*resetExecutionContext=*/false), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_reset_state(engine, /*resetExecutionContext=*/true), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, CallReturnsInvalidNullPointerForNullEngineOrName) {
    npu_vm_value arg = ov::intel_npu::vm::makeI64(1);
    EXPECT_EQ(npu_vm_call(nullptr, ENTRY_FUNC_NAME, 1, &arg), NPU_VM_ERROR_INVALID_NULL_POINTER);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_call(engine, nullptr, 1, &arg), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, CallReturnsFunctionNotFoundForMissingFunction) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_call(engine, MISSING_FUNC_NAME, 0, nullptr), NPU_VM_ERROR_FUNCTION_NOT_FOUND);
}

TEST_F(VirtualMachineApiTest, CallExecutesFunctionWithoutResults) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_call(engine, ENTRY_FUNC_NAME, 0, nullptr), NPU_VM_SUCCESS);
}

TEST_F(VirtualMachineApiTest, CallWithResultsReturnsInvalidNullPointerForNullEngineOrName) {
    const std::array<npu_vm_value, 2> args = {ov::intel_npu::vm::makeI64(1), ov::intel_npu::vm::makeI64(2)};
    npu_vm_value result = ov::intel_npu::vm::makeI64(0);
    EXPECT_EQ(npu_vm_call_with_results(nullptr, ADD_FUNC_NAME, static_cast<uint32_t>(args.size()), args.data(), 1, &result),
              NPU_VM_ERROR_INVALID_NULL_POINTER);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_call_with_results(engine, nullptr, static_cast<uint32_t>(args.size()), args.data(), 1, &result),
              NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, CallWithResultsReturnsFunctionCallFailedForSignatureMismatch) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);
    // Note: Function expects two arguments, but only one is provided
    const std::array<npu_vm_value, 1> args = {ov::intel_npu::vm::makeI64(1)};
    npu_vm_value result = ov::intel_npu::vm::makeI64(0);
    EXPECT_EQ(npu_vm_call_with_results(engine, ADD_FUNC_NAME, static_cast<uint32_t>(args.size()), args.data(), 1, &result),
              NPU_VM_ERROR_FUNCTION_CALL_FAILED);
}

TEST_F(VirtualMachineApiTest, CallWithResultsExecutesFunctionAndWritesResults) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);

    constexpr int64_t firstArg = 10;
    constexpr int64_t secondArg = 11;
    const std::array<npu_vm_value, 2> args = {ov::intel_npu::vm::makeI64(firstArg), ov::intel_npu::vm::makeI64(secondArg)};
    npu_vm_value result = ov::intel_npu::vm::makeI64(0);
    ASSERT_EQ(npu_vm_call_with_results(engine, ADD_FUNC_NAME, static_cast<uint32_t>(args.size()), args.data(), 1, &result), NPU_VM_SUCCESS);
    EXPECT_EQ(result.i64, firstArg + secondArg);
}

TEST_F(VirtualMachineApiTest, CallWithResultsSucceedsForTwoConsecutiveCallsWithDifferentArguments) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);

    {
        const std::array<npu_vm_value, 2> args = {ov::intel_npu::vm::makeI64(2), ov::intel_npu::vm::makeI64(3)};
        npu_vm_value result = ov::intel_npu::vm::makeI64(0);
        ASSERT_EQ(npu_vm_call_with_results(engine, ADD_FUNC_NAME, static_cast<uint32_t>(args.size()), args.data(), 1, &result),
                  NPU_VM_SUCCESS);
        EXPECT_EQ(result.i64, 5);
    }
    ASSERT_EQ(npu_vm_reset_state(engine, /*resetExecutionContext=*/false), NPU_VM_SUCCESS);
    {
        const std::array<npu_vm_value, 2> args = {ov::intel_npu::vm::makeI64(10), ov::intel_npu::vm::makeI64(20)};
        npu_vm_value result = ov::intel_npu::vm::makeI64(0);
        ASSERT_EQ(npu_vm_call_with_results(engine, ADD_FUNC_NAME, static_cast<uint32_t>(args.size()), args.data(), 1, &result),
                  NPU_VM_SUCCESS);
        EXPECT_EQ(result.i64, 30);
    }
}

TEST_F(VirtualMachineApiTest, CallWithResultsRoundTripsUnsignedValuesIncludingAboveSignedMax) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);

    const std::array<uint32_t, 5> values = {
            0U, 1U, static_cast<uint32_t>(INT32_MAX), static_cast<uint32_t>(INT32_MAX) + 1U, UINT32_MAX,
    };

    for (const auto value : values) {
        npu_vm_value arg{};
        arg.u32 = value;
        npu_vm_value result{};
        result.u32 = 0U;

        ASSERT_EQ(npu_vm_call_with_results(engine, UINT32_IDENTITY_FUNC_NAME, 1, &arg, 1, &result), NPU_VM_SUCCESS);
        EXPECT_EQ(result.u32, value);
    }
}

TEST_F(VirtualMachineApiTest, CallWithResultsSupportsBufferArgumentAndResult) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);

    std::array<uint8_t, 16> inputBuffer{};
    std::iota(inputBuffer.begin(), inputBuffer.end(), 0);  // Fill with sample data {0, 1, 2, ..., 15}
    npu_vm_value arg = ov::intel_npu::vm::makeBuffer(inputBuffer.data(), static_cast<uint32_t>(inputBuffer.size()));
    std::array<uint8_t, 16> outputBuffer{};  // Pre-allocated output buffer
    npu_vm_value result = ov::intel_npu::vm::makeBuffer(outputBuffer.data(), static_cast<uint32_t>(outputBuffer.size()));
    ASSERT_EQ(npu_vm_call_with_results(engine, BUFFER_IDENTITY_FUNC_NAME, 1, &arg, 1, &result), NPU_VM_SUCCESS);
    EXPECT_NE(result.buffer.data, nullptr);
    EXPECT_EQ(result.buffer.size, inputBuffer.size());
    EXPECT_TRUE(std::equal(inputBuffer.begin(), inputBuffer.end(), static_cast<uint8_t*>(result.buffer.data)));
}

TEST_F(VirtualMachineApiTest, CallWithResultsSupportsBufferArgumentAndInternallyAllocatedResult) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);

    std::array<uint8_t, 16> inputBuffer{};
    std::iota(inputBuffer.begin(), inputBuffer.end(), 0);  // Fill with sample data {0, 1, 2, ..., 15}
    npu_vm_value arg = ov::intel_npu::vm::makeBuffer(inputBuffer.data(), static_cast<uint32_t>(inputBuffer.size()));
    npu_vm_value result = ov::intel_npu::vm::makeBuffer(nullptr, 0);
    ASSERT_EQ(npu_vm_call_with_results(engine, BUFFER_IDENTITY_FUNC_NAME, 1, &arg, 1, &result), NPU_VM_SUCCESS);
    EXPECT_NE(result.buffer.data, nullptr);
    EXPECT_EQ(result.buffer.size, inputBuffer.size());
    EXPECT_TRUE(std::equal(inputBuffer.begin(), inputBuffer.end(), static_cast<uint8_t*>(result.buffer.data)));
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc)
    free(result.buffer.data);
}

TEST_F(VirtualMachineApiTest, CallWithResultsFailsIfPreAllocatedBufferResultIsTooSmall) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);

    std::array<uint8_t, 16> inputBuffer{};
    std::iota(inputBuffer.begin(), inputBuffer.end(), 0);  // Fill with sample data {0, 1, 2, ..., 15}
    npu_vm_value arg = ov::intel_npu::vm::makeBuffer(inputBuffer.data(), static_cast<uint32_t>(inputBuffer.size()));
    std::array<uint8_t, 8> outputBuffer{};  // Pre-allocated output buffer that is too small
    npu_vm_value result = ov::intel_npu::vm::makeBuffer(outputBuffer.data(), static_cast<uint32_t>(outputBuffer.size()));
    EXPECT_EQ(npu_vm_call_with_results(engine, BUFFER_IDENTITY_FUNC_NAME, 1, &arg, 1, &result),
              NPU_VM_ERROR_FUNCTION_CALL_FAILED);
}

TEST_F(VirtualMachineApiTest, CallWithResultsFailsIfBufferResultIsNullAndSizeIsNonZero) {
    const auto bytecode = makeValidBytecode();
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    ASSERT_EQ(npu_vm_load_module(engine, /*moduleParams*/ nullptr, module), NPU_VM_SUCCESS);

    std::array<uint8_t, 16> inputBuffer{};
    std::iota(inputBuffer.begin(), inputBuffer.end(), 0);  // Fill with sample data {0, 1, 2, ..., 15}
    npu_vm_value arg = ov::intel_npu::vm::makeBuffer(inputBuffer.data(), static_cast<uint32_t>(inputBuffer.size()));
    npu_vm_value result = ov::intel_npu::vm::makeBuffer(nullptr, 16);  // Invalid: null data pointer but non-zero size
    EXPECT_EQ(npu_vm_call_with_results(engine, BUFFER_IDENTITY_FUNC_NAME, 1, &arg, 1, &result),
              NPU_VM_ERROR_FUNCTION_CALL_FAILED);
}

TEST_F(VirtualMachineApiTest, InferReturnsInvalidNullPointerForNullEngineOrParams) {
    EXPECT_EQ(npu_vm_infer(nullptr, nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_infer(engine, nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

TEST_F(VirtualMachineApiTest, PredictOutputShapeReturnsInvalidNullPointerForNullEngineOrParams) {
    EXPECT_EQ(npu_vm_predict_output_shape(nullptr, nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
    ASSERT_EQ(npu_vm_new_engine(&engine), NPU_VM_SUCCESS);
    EXPECT_EQ(npu_vm_predict_output_shape(engine, nullptr), NPU_VM_ERROR_INVALID_NULL_POINTER);
}

}  // namespace
