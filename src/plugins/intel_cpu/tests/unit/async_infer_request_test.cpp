// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"

namespace {

class TestStreamsExecutor : public ov::threading::IStreamsExecutor {
public:
    explicit TestStreamsExecutor(int streams_num) : m_streams_num{streams_num} {}

    void run(ov::threading::Task task) override {
        std::thread worker([task = std::move(task)]() mutable {
            task();
        });
        worker.join();
    }

    void execute(ov::threading::Task task) override {
        task();
    }

    int get_stream_id() override {
        return 0;
    }

    int get_streams_num() override {
        return m_streams_num;
    }

    int get_numa_node_id() override {
        return 0;
    }

    int get_socket_id() override {
        return 0;
    }

    std::vector<int> get_rank() override {
        return {};
    }

    void cpu_reset() override {}

private:
    int m_streams_num = 0;
};

class TestInferRequest : public ov::IInferRequest {
public:
    void infer() override {
        infer_thread_id = std::this_thread::get_id();
        if (throw_on_infer) {
            throw std::runtime_error("infer failed");
        }
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>&) const override {
        return {};
    }

    void set_tensor(const ov::Output<const ov::Node>&, const ov::SoPtr<ov::ITensor>&) override {}

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>&) const override {
        return {};
    }

    void set_tensors(const ov::Output<const ov::Node>&, const std::vector<ov::SoPtr<ov::ITensor>>&) override {}

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override {
        return m_compiled_model;
    }

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override {
        return m_inputs;
    }

    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override {
        return m_outputs;
    }

    void check_tensors() const override {}

    bool throw_on_infer = false;
    std::thread::id infer_thread_id;

private:
    std::shared_ptr<const ov::ICompiledModel> m_compiled_model;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
};

TEST(AsyncInferRequestTest, UsesCallerThreadForSyncInferWhenRequested) {
    auto request = std::make_shared<TestInferRequest>();
    auto executor = std::make_shared<TestStreamsExecutor>(4);
    ov::intel_cpu::AsyncInferRequest async_request(request, executor, nullptr, false, true);

    const auto caller_thread_id = std::this_thread::get_id();
    async_request.infer();

    ASSERT_EQ(request->infer_thread_id, caller_thread_id);
}

TEST(AsyncInferRequestTest, KeepsExecutorDispatchForSyncInferWhenCallerThreadModeIsDisabled) {
    auto request = std::make_shared<TestInferRequest>();
    auto executor = std::make_shared<TestStreamsExecutor>(4);
    ov::intel_cpu::AsyncInferRequest async_request(request, executor, nullptr, false, false);

    const auto caller_thread_id = std::this_thread::get_id();
    async_request.infer();

    ASSERT_NE(request->infer_thread_id, caller_thread_id);
}

TEST(AsyncInferRequestTest, PropagatesExceptionsFromCallerThreadSyncInferPath) {
    auto request = std::make_shared<TestInferRequest>();
    auto executor = std::make_shared<TestStreamsExecutor>(4);
    ov::intel_cpu::AsyncInferRequest async_request(request, executor, nullptr, false, true);

    request->throw_on_infer = true;

    const auto caller_thread_id = std::this_thread::get_id();
    ASSERT_THROW(async_request.infer(), std::runtime_error);
    // Also pin the path: the throwing infer() must have run on the calling thread, otherwise
    // this would only assert that some exception surfaced from the regular dispatch path.
    ASSERT_EQ(request->infer_thread_id, caller_thread_id);
}

TEST(MultiAppThreadSyncExecutionConfigTest, KeepsRequestedValueEnabledForExplicitZeroStreams) {
    ov::intel_cpu::Config config;
    config.multiAppThreadSyncExecution = true;
    config.streams = 0;

    config.normalizeMultiAppThreadSyncExecution();

    ASSERT_TRUE(config.runSyncInferInCallerThread);
}

TEST(MultiAppThreadSyncExecutionConfigTest, DisablesCallerThreadExecutionForExclusiveAsyncRequests) {
    ov::intel_cpu::Config config;
    config.multiAppThreadSyncExecution = true;
    config.exclusiveAsyncRequests = true;

    config.normalizeMultiAppThreadSyncExecution();

    ASSERT_FALSE(config.runSyncInferInCallerThread);
}

TEST(MultiAppThreadSyncExecutionConfigTest, DisablesCallerThreadExecutionForSubStreams) {
    ov::intel_cpu::Config config;
    config.multiAppThreadSyncExecution = true;
    config.numSubStreams = 2;

    config.normalizeMultiAppThreadSyncExecution();

    ASSERT_FALSE(config.runSyncInferInCallerThread);
}

}  // namespace
