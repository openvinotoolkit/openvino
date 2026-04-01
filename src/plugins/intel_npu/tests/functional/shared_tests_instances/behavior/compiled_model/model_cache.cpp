// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/model_cache.hpp"

#include <common_test_utils/test_constants.hpp>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"

using namespace ov::test::behavior;

namespace {

std::vector<ov::AnyMap> config = {{{}}, {ov::enable_weightless(true)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         WeightlessCacheAccuracy,
                         ::testing::Combine(::testing::ValuesIn({true, false}),   // m_use_compile_model_api
                                            ::testing::Values(true),              // m_do_encryption
                                            ::testing::Values(ov::element::f16),  // m_inference_mode
                                            ::testing::Values(ov::element::f16),  // m_model_dtype
                                            ::testing::ValuesIn(config),          // config parsed with std::ignore
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),  // m_target_device
                         ov::test::utils::appendPlatformTypeTestName<WeightlessCacheAccuracy>);

}  // namespace
