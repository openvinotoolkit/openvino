// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compile_and_infer_different_element_type.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"


auto configs = []() {
    return std::vector<ov::AnyMap>{{{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                     ov::intel_npu::platform(ov::intel_npu::Platform::NPU3720),
                                     {"NPU_COMPILATION_MODE", "DefaultHW"}}},
                                   {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER),
                                     ov::intel_npu::platform(ov::intel_npu::Platform::NPU4000),
                                     {"NPU_COMPILATION_MODE", "DefaultHW"}}}};
};


INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, NPUInferRequestElementTypeTests,
        ::testing::Combine(::testing::Values(getFunction()),
                           ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                   {{1, 1, 128}, {1, 1, 128}}, {{128}, {128}}}),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::ValuesIn(configs())),
        ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);
