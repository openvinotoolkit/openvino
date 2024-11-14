// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/properties_tests.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;
namespace {

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         InferRequestPropertiesTest,
                         ::testing::Combine(::testing::Values(2u),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(std::vector<ov::AnyMap>{{}})),
                         ov::test::utils::appendPlatformTypeTestName<InferRequestPropertiesTest>);

}  // namespace
