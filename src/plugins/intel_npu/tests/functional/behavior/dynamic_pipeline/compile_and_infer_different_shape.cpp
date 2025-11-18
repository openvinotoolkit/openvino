// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/dynamic_pipeline/compile_and_infer_different_shape.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

const std::vector<std::string> models = {"MaxPool_canonical.xml", "CustomNet_canonical_strides_1x1_no_fork.xml"};

const std::vector<ov::AnyMap> configs = {{{"NPU_COMPILER_TYPE", "MLIR"},
                                          {"NPU_PLATFORM", "NPU4000"},
                                          {"NPU_COMPILATION_MODE", "HostCompile"},
                                          {"NPU_COMPILATION_MODE_PARAMS", "outline-entire-main-content=false"},
                                          {"COMPILATION_NUM_THREADS", 1},
                                          {"PERF_COUNT", "NO"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestDynamicShapeTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(models),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<InferRequestDynamicShapeTests>);
