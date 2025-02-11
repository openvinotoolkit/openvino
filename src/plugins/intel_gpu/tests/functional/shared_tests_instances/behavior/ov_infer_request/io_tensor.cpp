// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

auto emptyConfigs = []() {
    return std::vector<ov::AnyMap>{{}};
};

auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
        {ov::num_streams(ov::streams::AUTO)},
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
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

std::vector<ov::element::Type> supported_input_prcs = {
    ov::element::boolean,
    ov::element::f16,
    ov::element::f32,
    ov::element::f64,
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                 ::testing::ValuesIn(configs())),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(supported_input_prcs),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                 ::testing::ValuesIn(emptyConfigs())),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);
}  // namespace
