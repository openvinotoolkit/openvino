// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
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

const std::vector<ov::AnyMap> emptyConfigs = {{}};

const std::vector<ov::AnyMap> HeteroConfigs = {
            {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE)}};

const std::vector<ov::AnyMap> Multiconfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                 ::testing::ValuesIn(emptyConfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                 ::testing::ValuesIn(HeteroConfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                 ::testing::ValuesIn(Multiconfigs)),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);
}  // namespace
