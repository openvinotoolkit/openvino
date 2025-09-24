// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compile_and_infer_different_element_type.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

const std::vector<ov::AnyMap> configs = {};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestElementTypeTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<InferRequestElementTypeTests>);
