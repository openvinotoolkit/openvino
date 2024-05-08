// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/properties_tests.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "npu_private_properties.hpp"

using namespace ov::test::behavior;
namespace {

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, InferRequestPropertiesTest,
        ::testing::Combine(::testing::Values(2u), ::testing::Values(ov::test::utils::DEVICE_NPU),
                           ::testing::ValuesIn(std::vector<ov::AnyMap>{
                                   {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
                                    ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}})),
        ov::test::utils::appendPlatformTypeTestName<InferRequestPropertiesTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, InferRequestPropertiesTest,
                         ::testing::Combine(::testing::Values(2u), ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{{ov::intel_npu::compiler_type(
                                                    ov::intel_npu::CompilerType::DRIVER)}})),
                         ov::test::utils::appendPlatformTypeTestName<InferRequestPropertiesTest>);
}  // namespace
