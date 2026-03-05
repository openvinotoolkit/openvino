// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/plugin/test_properties.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"

using namespace ov::test::behavior;

const std::vector<std::string> supported_configs = {{ov::hint::performance_mode.name()},
                                                    {ov::cache_dir.name()},
                                                    {ov::intel_npu::driver_version.name()}};
const std::vector<std::string> supported_compiler_configs = {{ov::intel_npu::turbo.name()},
                                                             {ov::intel_npu::qdq_optimization.name()}};
const std::vector<std::string> unsupported_compiler_configs = {{"DUMMY_PROPERTY"}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         PropertiesManagerTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(supported_configs)),
                         PropertiesManagerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ExpectLoadingCompilerPropertySupported,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(supported_compiler_configs)),
                         PropertiesManagerTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         ExpectLoadingCompilerPropertyNotSupported,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(unsupported_compiler_configs)),
                         PropertiesManagerTests::getTestCaseName);
