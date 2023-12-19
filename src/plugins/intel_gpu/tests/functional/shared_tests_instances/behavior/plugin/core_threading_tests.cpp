// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior/plugin/core_threading.hpp>
#include "remote_tensor_tests/helpers.hpp"
#include "common_test_utils/subgraph_builders/split_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/single_conv.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "common_test_utils/subgraph_builders/2_input_subtract.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;

namespace {

auto params = []() {
    return std::vector<Params>{
        std::tuple<Device, Config>{ov::test::utils::DEVICE_GPU, {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}}},
        std::tuple<Device, Config>{ov::test::utils::DEVICE_GPU, {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(NO)}}},
        std::tuple<Device, Config>{ov::test::utils::DEVICE_GPU, {{CONFIG_KEY(CACHE_DIR), "cache"}}},
    };
};

}  // namespace

// tested function: CreateContext, LoadNetwork, AddExtension
TEST_P(CoreThreadingTestsWithIterations, smoke_LoadNetwork_RemoteContext) {
    InferenceEngine::Core ie;
    std::atomic<unsigned int> counter{0u};

    std::vector<InferenceEngine::CNNNetwork> networks;
    networks.emplace_back(InferenceEngine::CNNNetwork(ov::test::utils::make_2_input_subtract()));
    networks.emplace_back(InferenceEngine::CNNNetwork(ov::test::utils::make_multi_single_conv()));
    networks.emplace_back(InferenceEngine::CNNNetwork(ov::test::utils::make_single_conv()));
    networks.emplace_back(InferenceEngine::CNNNetwork(ov::test::utils::make_split_conv_concat()));
    networks.emplace_back(InferenceEngine::CNNNetwork(ov::test::utils::make_split_multi_conv_concat()));

    auto ocl_instance = std::make_shared<OpenCL>();
    ie.SetConfig(config, target_device);
    runParallel([&] () {
        auto value = counter++;
        auto remote_context = make_shared_context(ie, ov::test::utils::DEVICE_GPU, ocl_instance->_context.get());
        (void)ie.LoadNetwork(networks[value % networks.size()], remote_context);
    }, numIterations, numThreads);
}

INSTANTIATE_TEST_SUITE_P(smoke_GPU, CoreThreadingTests, testing::ValuesIn(params()), CoreThreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPU, CoreThreadingTestsWithIterations,
    testing::Combine(testing::ValuesIn(params()),
                     testing::Values(4),
                     testing::Values(20),
                     testing::Values(ModelClass::Default)),
    CoreThreadingTestsWithIterations::getTestCaseName);
