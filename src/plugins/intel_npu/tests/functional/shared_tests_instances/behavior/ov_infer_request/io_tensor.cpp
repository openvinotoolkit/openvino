//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "overload/ov_infer_request/io_tensor.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> pluginCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}};

const std::vector<ov::AnyMap> pluginCompilerMultiConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
         ov::device::priorities(ov::test::utils::DEVICE_NPU)}};

const std::vector<ov::AnyMap> pluginCompilerAutoConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
         ov::device::priorities(ov::test::utils::DEVICE_NPU)}};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

const std::vector<ov::AnyMap> driverCompilerMultiConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
         ov::device::priorities(ov::test::utils::DEVICE_NPU)}};

const std::vector<ov::AnyMap> driverCompilerAutoConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
         ov::device::priorities(ov::test::utils::DEVICE_NPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(pluginCompilerMultiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestIOTensorTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(pluginCompilerMultiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests_Driver, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(driverCompilerMultiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestIOTensorTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests_Driver, OVInferRequestIOTensorTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(driverCompilerMultiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_Driver, OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

const std::vector<ov::element::Type> prcs = {
        ov::element::boolean, ov::element::bf16, ov::element::f16, ov::element::f32, ov::element::f64, ov::element::i4,
        ov::element::i8,      ov::element::i16,  ov::element::i32, ov::element::i64, ov::element::u1,  ov::element::u4,
        ov::element::u8,      ov::element::u16,  ov::element::u32, ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(pluginCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(pluginCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorSetPrecisionTestNPU,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTestNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_Mutli_BehaviorTests, OVInferRequestIOTensorSetPrecisionTestNPU,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(pluginCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTestNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestIOTensorSetPrecisionTestNPU,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(pluginCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTestNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pluginCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestCheckTensorPrecision>);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(pluginCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestCheckTensorPrecision>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(pluginCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestCheckTensorPrecision>);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests_Driver, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(driverCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_Driver, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(driverCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestIOTensorSetPrecisionTestNPU,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTestNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_Mutli_BehaviorTests_Driver, OVInferRequestIOTensorSetPrecisionTestNPU,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(driverCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTestNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_Driver, OVInferRequestIOTensorSetPrecisionTestNPU,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(driverCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestIOTensorSetPrecisionTestNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestCheckTensorPrecision>);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests_Driver, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(driverCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestCheckTensorPrecision>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_Driver, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs), ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(driverCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestCheckTensorPrecision>);
}  // namespace
