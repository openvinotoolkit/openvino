// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/backend/zero_tensor_tests.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configsInferRequestRunTests = {{}};
const std::vector<ov::element::Type> defaultTensorDataType = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ZeroTensorTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests),
                                            ::testing::ValuesIn(defaultTensorDataType)),
                         ZeroTensorTests::getTestCaseName);

const std::vector<ov::element::Type> supportedTensorDataTypes = {
    ov::element::f32,    ov::element::f16, ov::element::bf16, ov::element::f8e4m3, ov::element::f8e5m2,
    ov::element::f8e8m0, ov::element::nf4, ov::element::u2,   ov::element::u4,     ov::element::i4,
    ov::element::u8,     ov::element::i8,  ov::element::u16,  ov::element::i16,    ov::element::u32,
    ov::element::i32,    ov::element::u64, ov::element::i64,  ov::element::f64,    ov::element::boolean,
};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTest,
                         ZeroTensorTestsCheckDataType,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configsInferRequestRunTests),
                                            ::testing::ValuesIn(supportedTensorDataTypes)),
                         ZeroTensorTestsCheckDataType::getTestCaseName);
