// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/io_tensor.hpp"

using namespace ov::test::behavior;

namespace {

auto emptyConfigs = []() {
    return std::vector<ov::AnyMap>{{}};
};

auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
        {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS,
          InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}},
    };
};

auto AutoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_GPU) + "(4)"},
         // no timeout to avoid increasing the test time
         {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestIOTensorTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                        OVInferRequestIOTensorTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestIOTensorTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs())),
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

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestIOTensorSetPrecisionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(prcs),
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs())),
                         OVInferRequestIOTensorSetPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(supported_input_prcs),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                 ::testing::ValuesIn(emptyConfigs())),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestCheckTensorPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(supported_input_prcs),
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs())),
                         OVInferRequestCheckTensorPrecision::getTestCaseName);
}  // namespace
