// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_threading.hpp"

#include <condition_variable>
#include <mutex>

#include "openvino/runtime/intel_cpu/properties.hpp"

namespace {
const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_CPU, {{ov::enable_profiling(true)}}},
};

const Params paramsStreams[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_CPU, {{ov::num_streams(ov::streams::AUTO)}}},
};

// Forces application threads to call infer() at the same instant, proving genuine concurrent
// overlap. Plain mutex/condition_variable + generation counter since this targets C++17.
class LocalThreadBarrier {
public:
    explicit LocalThreadBarrier(unsigned expected_count) : m_expected(expected_count) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        const unsigned generation = m_generation;
        if (++m_waiting == m_expected) {
            m_waiting = 0;
            ++m_generation;
            m_cv.notify_all();
        } else {
            m_cv.wait(lock, [&] {
                return m_generation != generation;
            });
        }
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    unsigned m_expected;
    unsigned m_waiting = 0;
    unsigned m_generation = 0;
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
        // CPU pinning is disabled: this property rejects pinning resolved to true, which isn't
        // what this concurrent-overlap test targets.
        compileConfig[ov::hint::enable_cpu_pinning.name()] = false;

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

        // Warm-up: absorb first-call lazy init cost before the timed/compared runs below.
        for (unsigned int i = 0; i < numThreads; ++i) {
            requests[i].infer();
        }

        auto inferViaThread = [&](unsigned int i, LocalThreadBarrier& barrier) {
            // Blocks until all threads arrive, so infer() calls launch in genuine overlap,
            // independent of noisy wall-clock timing.
            barrier.arrive_and_wait();
            requests[i].infer();
            const auto output = requests[i].get_output_tensor(0);
            outputs[i] = ov::Tensor(output.get_element_type(), output.get_shape());
            output.copy_to(outputs[i]);
        };

        LocalThreadBarrier barrier(numThreads);
        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        for (unsigned int i = 0; i < numThreads; ++i) {
            threads.emplace_back(inferViaThread, i, std::ref(barrier));
        }
        for (auto& thread : threads) {
            thread.join();
        }

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