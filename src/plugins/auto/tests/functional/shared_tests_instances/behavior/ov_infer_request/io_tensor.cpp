// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/io_tensor.hpp"

#include <vector>

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> Autoconfigs = {{ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}};

const std::vector<ov::AnyMap> emptyConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

std::vector<ov::element::Type> prcs = {
    ov::element::boolean,
    ov::element::bf16,
    ov::element::f16,
    ov::element::f32,
    ov::element::f64,
    ov::element::i4,
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u1,
    ov::element::u4,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64,
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);
}  // namespace
