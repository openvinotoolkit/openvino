// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/io_tensor.hpp"

#include <vector>

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

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

std::vector<ov::element::Type> supported_input_prcs = {ov::element::f32, ov::element::i16, ov::element::u8};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestIOTensorTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(::testing::ValuesIn(supported_input_prcs),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

}  // namespace
