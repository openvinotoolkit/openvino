//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_artifacts.hpp"
#include "vm_runtime/virtual_machine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace {

struct ArtifactTestCase {
    std::string name;
};

double bitsToF64(int64_t bits) {
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int64_t f64ToBits(double value) {
    int64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

class ScopedModule final {
public:
    ScopedModule(const ScopedModule&) = delete;
    ScopedModule& operator=(const ScopedModule&) = delete;
    ScopedModule(ScopedModule&&) = delete;
    ScopedModule& operator=(ScopedModule&&) = delete;

    explicit ScopedModule(npu_vm_module* rawModule): _module(rawModule) {
    }

    ~ScopedModule() {
        if (_module != nullptr) {
            npu_vm_destroy_module(_module);
        }
    }

    npu_vm_module* get() const {
        return _module;
    }

private:
    npu_vm_module* _module = nullptr;
};

class ScopedEngine final {
public:
    ScopedEngine(const ScopedEngine&) = delete;
    ScopedEngine& operator=(const ScopedEngine&) = delete;
    ScopedEngine(ScopedEngine&&) = delete;
    ScopedEngine& operator=(ScopedEngine&&) = delete;

    explicit ScopedEngine(npu_vm_engine* rawEngine): _engine(rawEngine) {
    }

    ~ScopedEngine() {
        if (_engine != nullptr) {
            npu_vm_destroy_engine(_engine);
        }
    }

    npu_vm_engine* get() const {
        return _engine;
    }

private:
    npu_vm_engine* _engine = nullptr;
};

class ScopedFunctionInfo final {
public:
    ScopedFunctionInfo(const ScopedFunctionInfo&) = delete;
    ScopedFunctionInfo& operator=(const ScopedFunctionInfo&) = delete;
    ScopedFunctionInfo(ScopedFunctionInfo&&) = delete;
    ScopedFunctionInfo& operator=(ScopedFunctionInfo&&) = delete;

    explicit ScopedFunctionInfo(npu_vm_function_info* rawInfo): _info(rawInfo) {
    }

    ~ScopedFunctionInfo() {
        if (_info != nullptr) {
            npu_vm_destroy_function_info(_info);
        }
    }

    npu_vm_function_info* get() const {
        return _info;
    }

private:
    npu_vm_function_info* _info = nullptr;
};

std::vector<ArtifactTestCase> getArtifactTestCases() {
    std::vector<ArtifactTestCase> cases;
    cases.reserve(BYTECODE_ARTIFACTS.size());
    for (const auto& artifact : BYTECODE_ARTIFACTS) {
        cases.push_back({artifact.first});
    }
    std::sort(cases.begin(), cases.end(), [](const ArtifactTestCase& a, const ArtifactTestCase& b) {
        return a.name < b.name;
    });
    return cases;
}

class VirtualMachineBackwardCompatibilityTest : public testing::TestWithParam<ArtifactTestCase> {};

TEST_P(VirtualMachineBackwardCompatibilityTest, ValidateArtifactAgainstCurrentVm) {
    const auto& testCase = GetParam();
    const auto artifactIt = BYTECODE_ARTIFACTS.find(testCase.name);
    ASSERT_TRUE(artifactIt != BYTECODE_ARTIFACTS.end()) << "Artifact not found: " << testCase.name;
    const auto& artifact = artifactIt->second;

    const auto& bytecode = artifact.binaryArtifact;
    ASSERT_FALSE(bytecode.empty()) << "Empty bytecode artifact for: " << testCase.name;

    if (artifact.testType == TestType::ParseFailure) {
        npu_vm_module* rawInvalidModule = nullptr;
        const auto parseResult =
                npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &rawInvalidModule);
        if (parseResult == NPU_VM_SUCCESS && rawInvalidModule != nullptr) {
            npu_vm_destroy_module(rawInvalidModule);
        }
        EXPECT_NE(parseResult, NPU_VM_SUCCESS) << "Expected parse failure for artifact: " << testCase.name;
        return;
    }

    npu_vm_module* rawModule = nullptr;
    ASSERT_EQ(npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &rawModule), NPU_VM_SUCCESS)
            << "Failed to parse artifact: " << testCase.name;
    ScopedModule module(rawModule);

    npu_vm_engine* rawEngine = nullptr;
    ASSERT_EQ(npu_vm_new_engine(&rawEngine), NPU_VM_SUCCESS);
    ScopedEngine engine(rawEngine);

    ASSERT_EQ(npu_vm_load_module(engine.get(), /*moduleParams*/ nullptr, module.get()), NPU_VM_SUCCESS)
            << "Failed to load artifact module: " << testCase.name;

    if (artifact.testType == TestType::ParseSuccess) {
        SUCCEED();
        return;
    }

    ASSERT_TRUE(artifact.testType == TestType::Accurate || artifact.testType == TestType::RuntimeFailure)
            << "Unknown test type for artifact " << testCase.name;

    const auto functionName = artifact.entryPointFuncName;
    ASSERT_FALSE(functionName.empty()) << "Empty entry point function name for artifact: " << testCase.name;

    npu_vm_function_info* rawInfo = nullptr;
    ASSERT_EQ(npu_vm_get_function_info(module.get(), functionName.c_str(), &rawInfo), NPU_VM_SUCCESS)
            << "Failed to get function info for artifact function: " << functionName;
    ScopedFunctionInfo functionInfo(rawInfo);

    const auto* info = functionInfo.get();
    ASSERT_NE(info, nullptr);
    std::vector<npu_vm_type> paramTypes;
    paramTypes.reserve(info->num_params);
    std::copy_n(info->param_types, info->num_params, std::back_inserter(paramTypes));

    std::vector<npu_vm_type> resultTypes;
    resultTypes.reserve(info->num_results);
    std::copy_n(info->result_types, info->num_results, std::back_inserter(resultTypes));

    ASSERT_EQ(artifact.inputValues.size(), static_cast<size_t>(info->num_params))
            << "Input value count mismatch for artifact " << testCase.name << ": expected " << info->num_params
            << ", got " << artifact.inputValues.size();

    std::vector<npu_vm_value> arguments;
    arguments.reserve(info->num_params);

    for (uint32_t index = 0; index < info->num_params; ++index) {
        const auto& input = artifact.inputValues.at(index);
        const auto type = paramTypes.at(index);

        npu_vm_value arg{};
        if (type == npu_vm_type_float64) {
            arg.f64 = std::holds_alternative<double>(input) ? std::get<double>(input)
                                                            : bitsToF64(std::get<int64_t>(input));
        } else {
            arg.i64 = std::holds_alternative<int64_t>(input) ? std::get<int64_t>(input)
                                                             : static_cast<int64_t>(std::get<double>(input));
        }
        arguments.push_back(arg);
    }

    std::vector<npu_vm_value> results(info->num_results);
    for (uint32_t index = 0; index < info->num_results; ++index) {
        if (resultTypes.at(index) == npu_vm_type_float64) {
            results.at(index).f64 = 0.0;
        } else {
            results.at(index).i64 = 0;
        }
    }

    if (arguments.size() > std::numeric_limits<uint32_t>::max() ||
        results.size() > std::numeric_limits<uint32_t>::max()) {
        FAIL() << "Input or output size exceeds maximum allowed for artifact " << testCase.name;
    }
    const auto vmCallResult =
            npu_vm_call_with_results(engine.get(), functionName.c_str(), static_cast<uint32_t>(arguments.size()),
                                     arguments.data(), static_cast<uint32_t>(results.size()), results.data());

    if (artifact.testType == TestType::RuntimeFailure) {
        EXPECT_NE(vmCallResult, NPU_VM_SUCCESS) << "Expected runtime failure for artifact " << testCase.name
                                                << ", but execution completed successfully.";
        return;
    }

    ASSERT_EQ(vmCallResult, NPU_VM_SUCCESS) << "VM execution failed for artifact " << testCase.name;

    ASSERT_EQ(artifact.expectedValues.size(), static_cast<size_t>(info->num_results))
            << "Expected value count mismatch for artifact " << testCase.name << ": expected " << info->num_results
            << ", got " << artifact.expectedValues.size();

    for (uint32_t index = 0; index < info->num_results; ++index) {
        const auto& expected = artifact.expectedValues.at(index);
        const auto type = resultTypes.at(index);

        if (type == npu_vm_type_float64) {
            const double expectedF64 = std::holds_alternative<double>(expected)
                                               ? std::get<double>(expected)
                                               : bitsToF64(std::get<int64_t>(expected));
            EXPECT_DOUBLE_EQ(results.at(index).f64, expectedF64)
                    << "Output mismatch at index " << index << " in artifact " << testCase.name;
            continue;
        }

        const int64_t expectedI64 = std::holds_alternative<int64_t>(expected) ? std::get<int64_t>(expected)
                                                                              : f64ToBits(std::get<double>(expected));
        EXPECT_EQ(results.at(index).i64, expectedI64)
                << "Output mismatch at index " << index << " in artifact " << testCase.name;
    }
}

std::string getArtifactTestName(const testing::TestParamInfo<ArtifactTestCase>& paramInfo) {
    std::string name = paramInfo.param.name;
    for (auto& character : name) {
        if (!std::isalnum(static_cast<unsigned char>(character))) {
            character = '_';
        }
    }
    return name;
}

std::ostream& operator<<(std::ostream& os, const ArtifactTestCase& params) {
    return os << params.name;
}

INSTANTIATE_TEST_SUITE_P(BytecodeArtifactCompatibility, VirtualMachineBackwardCompatibilityTest,
                         testing::ValuesIn(getArtifactTestCases()), getArtifactTestName);

}  // namespace
