// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <thread>

#include "behavior/plugin/auto_batching_tests.hpp"
#include "behavior/plugin/configuration_tests.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/relu.hpp"

const std::vector<size_t> num_streams{ 2 };
const std::vector<bool>   get_vs_set{ true, false };
const std::vector<size_t> num_requests{ 1, 8, 16, 64 };
const std::vector<size_t> num_batch{ 1, 8, 32, 256 };
using namespace AutoBatchingTests;
using namespace BehaviorTestsDefinitions;

namespace AutoBatchingTests {

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_GPU, AutoBatching_Test,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                 ::testing::ValuesIn(get_vs_set),
                                 ::testing::ValuesIn(num_streams),
                                 ::testing::ValuesIn(num_requests),
                                 ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatching_GPU, AutoBatching_Test_DetectionOutput,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                 ::testing::ValuesIn(get_vs_set),
                                 ::testing::ValuesIn(num_streams),
                                 ::testing::ValuesIn(num_requests),
                                 ::testing::ValuesIn(num_batch)),
                         AutoBatching_Test_DetectionOutput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_AutoBatching_GPU,
        DefaultConfigurationTest,
        ::testing::Combine(
                ::testing::Values(std::string(CommonTestUtils::DEVICE_BATCH) + ":" + CommonTestUtils::DEVICE_GPU),
                ::testing::Values(DefaultParameter{CONFIG_KEY(AUTO_BATCH_TIMEOUT),
                                                   InferenceEngine::Parameter{"1000"}})),
        DefaultConfigurationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_AutoBatching_GPU_2_0_string,
        DefaultConfigurationTest,
        ::testing::Combine(
                ::testing::Values(std::string(CommonTestUtils::DEVICE_BATCH) + ":" + CommonTestUtils::DEVICE_GPU),
                ::testing::Values(DefaultParameter{ov::auto_batch_timeout.name(),
                                                   InferenceEngine::Parameter{"1000"}})),
        DefaultConfigurationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_AutoBatching_GPU_2_0_uint,
        DefaultConfigurationTest,
        ::testing::Combine(
                ::testing::Values(std::string(CommonTestUtils::DEVICE_BATCH) + ":" + CommonTestUtils::DEVICE_GPU),
                ::testing::Values(DefaultParameter{ov::auto_batch_timeout.name(),
                                                   InferenceEngine::Parameter{1000}})),
        DefaultConfigurationTest::getTestCaseName);

using namespace ov;

class AutoBatchCompileThreading : public testing::Test {
public:
    static void runParallel(std::function<void(void)> func,
                            const unsigned int iterations = 50,
                            const unsigned int threadsNum = 24) {
        std::vector<std::thread> threads(threadsNum);
        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }
        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }
};

TEST_F(AutoBatchCompileThreading, AutoBatchCloningThreadSafetyTest) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 3, 4});
    ov::Output<Node> intermediate = input->output(0);
    for (size_t i = 0; i < 100; ++i)
        intermediate = std::make_shared<op::v0::Relu>(input)->output(0);
    auto output = std::make_shared<op::v0::Result>(intermediate);
    auto model = std::make_shared<ov::Model>(ResultVector{output}, ParameterVector{input});
    auto core = ov::Core();
    runParallel([&]() {
        core.compile_model(model, "GPU");
    });
}
}  // namespace AutoBatchingTests
