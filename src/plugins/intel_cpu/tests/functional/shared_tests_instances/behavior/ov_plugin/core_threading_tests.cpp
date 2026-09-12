// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_threading.hpp"

#include "openvino/runtime/intel_cpu/properties.hpp"

namespace {
const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_CPU, {{ov::enable_profiling(true)}}},
};

const Params paramsStreams[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_CPU, {{ov::num_streams(ov::streams::AUTO)}}},
};
}  // namespace

class CoreThreadingCpuMultiAppThreadSyncTest : public CoreThreadingTest {};

// tested function: compile_model, create_infer_request, infer
TEST_P(CoreThreadingCpuMultiAppThreadSyncTest, smoke_CpuExecNetworkMultiAppThreadSyncWithStreams) {
    auto core = ov::test::utils::create_core();
    core.set_property(target_device, config);

    constexpr unsigned int numThreads = 4;

    auto runInferWithStreams = [&](bool syncExec) -> std::vector<ov::Tensor> {
        Config compileConfig = config;
        compileConfig[ov::intel_cpu::multi_app_thread_sync_execution.name()] = syncExec;
        compileConfig[ov::num_streams.name()] = static_cast<int32_t>(numThreads);

        auto compiledModel = core.compile_model(ov::test::utils::make_single_conv(), target_device, compileConfig);
        EXPECT_EQ(static_cast<int32_t>(numThreads), compiledModel.get_property(ov::num_streams));

        std::vector<ov::InferRequest> requests;
        std::vector<ov::Tensor> inputTensors;
        std::vector<ov::Tensor> outputs(numThreads);
        requests.reserve(numThreads);
        inputTensors.reserve(numThreads);

        const auto input = compiledModel.input();
        for (unsigned int i = 0; i < numThreads; ++i) {
            requests.push_back(compiledModel.create_infer_request());
            inputTensors.push_back(ov::test::utils::create_and_fill_tensor(input.get_element_type(),
                                                                           input.get_shape(),
                                                                           256,
                                                                           static_cast<double_t>(i + 1)));
            requests.back().set_input_tensor(inputTensors.back());
        }

        CoreThreadingTestsBase::runParallelIndexed(
            [&](size_t i) {
                requests[i].infer();
                const auto output = requests[i].get_output_tensor(0);
                outputs[i] = ov::Tensor(output.get_element_type(), output.get_shape());
                output.copy_to(outputs[i]);
            },
            numThreads);

        return outputs;
    };

    const auto outFalse = runInferWithStreams(false);
    const auto outTrue = runInferWithStreams(true);

    ASSERT_EQ(outFalse.size(), outTrue.size());
    for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        ASSERT_GT(outFalse[threadIndex].get_size(), 0u);
        ASSERT_GT(outTrue[threadIndex].get_size(), 0u);
        ov::test::utils::compare(outFalse[threadIndex], outTrue[threadIndex]);
    }
}

INSTANTIATE_TEST_SUITE_P(CPU, CoreThreadingTest, testing::ValuesIn(params), CoreThreadingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU,
                         CoreThreadingCpuMultiAppThreadSyncTest,
                         testing::ValuesIn(params),
                         CoreThreadingCpuMultiAppThreadSyncTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(params), testing::Values(4), testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU_Streams,
                         CoreThreadingTestsWithCacheEnabled,
                         testing::Combine(testing::ValuesIn(paramsStreams), testing::Values(20), testing::Values(10)),
                         CoreThreadingTestsWithCacheEnabled::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CPU_Streams,
                         CoreThreadingTestsWithIter,
                         testing::Combine(testing::ValuesIn(paramsStreams), testing::Values(4), testing::Values(50)),
                         CoreThreadingTestsWithIter::getTestCaseName);