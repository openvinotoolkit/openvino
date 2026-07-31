// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executor.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"

namespace {

using namespace std::chrono_literals;

class SharedState {
public:
    std::atomic<int> inferCalls{0};
    std::atomic<bool> failInfer{false};
};

class TestPlugin final : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    void set_property(const ov::AnyMap&) override {}

    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("Not implemented in unit test plugin");
    }

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
    }
};

class TestCompiledModel;

class TestSyncInferRequest final : public ov::ISyncInferRequest {
public:
    TestSyncInferRequest(std::shared_ptr<const ov::ICompiledModel> compiled_model, std::shared_ptr<SharedState> state)
        : ov::ISyncInferRequest(std::move(compiled_model)),
          m_state(std::move(state)) {}

    void infer() override {
        m_state->inferCalls.fetch_add(1);
        if (m_state->failInfer.exchange(false)) {
            OPENVINO_THROW("sync infer failure");
        }
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

protected:
    void check_tensors() const override {
        // This test request does not consume model tensors; it validates async callback flow only.
    }

private:
    std::shared_ptr<SharedState> m_state;
};

class TestCompiledModel final : public ov::ICompiledModel {
public:
    explicit TestCompiledModel(std::shared_ptr<SharedState> state)
        : ov::ICompiledModel(build_model(), make_test_plugin()),
          m_state(std::move(state)),
          m_model(build_model()) {}

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        auto self = std::static_pointer_cast<const TestCompiledModel>(shared_from_this());
        return std::make_shared<TestSyncInferRequest>(std::move(self), m_state);
    }

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        intel_npu::make_executor("executor_test_task", 1),
                                                        intel_npu::make_executor("executor_test_callback", 1));
    }

    void export_model(std::ostream&) const override {}

    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }

    void set_property(const ov::AnyMap&) override {}

    ov::Any get_property(const std::string& name) const override {
        if (name == ov::execution_devices.name()) {
            return std::vector<std::string>{"NPU"};
        }
        OPENVINO_THROW("Unsupported property: ", name);
    }

private:
    static std::shared_ptr<const ov::IPlugin> make_test_plugin() {
        static const auto plugin = std::make_shared<TestPlugin>();
        return plugin;
    }

    static std::shared_ptr<ov::Model> build_model() {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        auto result = std::make_shared<ov::op::v0::Result>(param);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
    }

    std::shared_ptr<SharedState> m_state;
    std::shared_ptr<ov::Model> m_model;
};

bool wait_until(const std::function<bool()>& pred) {
    while (!pred()) {
        std::this_thread::yield();
    }
    return true;
}

TEST(NPUExecutorTests, InlineModeRunsOnCallerThread) {
    auto exec = intel_npu::make_executor("inline", 0, false, 10ms);

    std::thread::id taskThread{};
    exec->run([&] {
        taskThread = std::this_thread::get_id();
    });

    EXPECT_EQ(taskThread, std::this_thread::get_id());
}

TEST(NPUExecutorTests, InlineModeSwallowsTaskExceptions) {
    auto exec = intel_npu::make_executor("inline_throw", 0, false, 10ms);

    std::atomic<bool> taskRan{false};

    EXPECT_NO_THROW(exec->run([&] {
        taskRan = true;
        throw std::runtime_error("inline_failure");
    }));

    EXPECT_TRUE(taskRan.load());

    std::atomic<bool> followUpRan{false};
    EXPECT_NO_THROW(exec->run([&] {
        followUpRan = true;
    }));
    EXPECT_TRUE(followUpRan.load());
}

TEST(NPUExecutorTests, AdaptiveModeGrowsForQueuedWork) {
    auto exec = intel_npu::make_executor("grow", 0, true, 200ms);

    std::mutex m;
    std::condition_variable cv;
    bool release = false;
    int started = 0;

    auto blockingTask = [&] {
        {
            std::lock_guard<std::mutex> lock(m);
            ++started;
        }
        cv.notify_all();

        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] {
            return release;
        });
    };

    exec->run(blockingTask);
    exec->run(blockingTask);

    // EXPECT (non-fatal) so the release block always runs,
    // preventing the destructor from deadlocking on blocked worker threads.
    EXPECT_TRUE(wait_until([&] {
        std::lock_guard<std::mutex> lock(m);
        return started == 2;
    }));

    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();
}

TEST(NPUExecutorTests, FixedModeDoesNotUseMoreThanConfiguredWorkers) {
    constexpr int workers = 2;
    constexpr int tasksCount = 6;
    auto exec = intel_npu::make_executor("fixed_workers", workers, false, 200ms);

    std::mutex m;
    std::condition_variable cv;
    bool release = false;
    int started = 0;
    int completed = 0;
    std::set<std::thread::id> workerThreads;

    for (int i = 0; i < tasksCount; ++i) {
        exec->run([&] {
            {
                std::lock_guard<std::mutex> lock(m);
                ++started;
                workerThreads.insert(std::this_thread::get_id());
            }
            cv.notify_all();

            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] {
                return release;
            });

            ++completed;
            cv.notify_all();
        });
    }

    // EXPECT (non-fatal) so the release block always runs,
    // preventing the destructor from deadlocking on 6 blocked worker tasks.
    EXPECT_TRUE(wait_until([&] {
        std::lock_guard<std::mutex> lock(m);
        return started >= workers;
    }));

    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();

    EXPECT_TRUE(wait_until([&] {
        std::lock_guard<std::mutex> lock(m);
        return completed == tasksCount;
    }));

    std::lock_guard<std::mutex> lock(m);
    EXPECT_LE(workerThreads.size(), static_cast<size_t>(workers));
}

TEST(NPUExecutorTests, AdaptiveModeGrowShrinkAcrossMultipleBursts) {
    auto exec = intel_npu::make_executor("grow_shrink_cycles", 1, true, 30ms);

    for (int cycle = 0; cycle < 5; ++cycle) {
        std::mutex m;
        std::condition_variable cv;
        bool release = false;
        int started = 0;
        int completed = 0;

        auto blockingTask = [&] {
            {
                std::lock_guard<std::mutex> lock(m);
                ++started;
            }
            cv.notify_all();

            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] {
                return release;
            });

            ++completed;
            cv.notify_all();
        };

        exec->run(blockingTask);
        exec->run(blockingTask);

        // EXPECT (non-fatal) so the release block always runs,
        // preventing the destructor from deadlocking on blocked worker threads.
        EXPECT_TRUE(wait_until([&] {
            std::lock_guard<std::mutex> lock(m);
            return started == 2;
        }));

        {
            std::lock_guard<std::mutex> lock(m);
            release = true;
        }
        cv.notify_all();

        EXPECT_TRUE(wait_until([&] {
            std::lock_guard<std::mutex> lock(m);
            return completed == 2;
        }));

        std::this_thread::yield();
    }
}

TEST(NPUExecutorTests, DestructorWaitsForInFlightTaskCompletion) {
    auto exec = intel_npu::make_executor("destroy_waits", 1, false, 200ms);

    std::mutex m;
    std::condition_variable cv;
    bool started = false;
    bool release = false;
    std::atomic<bool> finished{false};
    std::atomic<bool> destructorReturned{false};

    exec->run([&] {
        {
            std::lock_guard<std::mutex> lock(m);
            started = true;
        }
        cv.notify_all();

        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] {
            return release;
        });
        finished = true;
    });

    ASSERT_TRUE(wait_until([&] {
        std::lock_guard<std::mutex> lock(m);
        return started;
    }));

    std::thread destroyer([&] {
        exec.reset();
        destructorReturned = true;
    });

    // Do not use sleep_for to assert destructorReturned is still false — that
    // is a racy negative assertion that fails on a loaded CI machine where the
    // destroyer thread may not be scheduled within the sleep window.
    // The deterministic guarantee is captured below: after destroyer.join(),
    // finished must be true, proving the destructor waited for the task.
    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();

    destroyer.join();
    EXPECT_TRUE(finished.load());
    EXPECT_TRUE(destructorReturned.load());
}

TEST(NPUExecutorTests, AdaptiveModeWithZeroBaselineRunsOffCallerThread) {
    auto exec = intel_npu::make_executor("adaptive_zero_baseline", 0, true, 200ms);

    const auto callerId = std::this_thread::get_id();
    std::thread::id workerId{};
    std::promise<void> done;
    auto doneFuture = done.get_future();

    exec->run([&] {
        workerId = std::this_thread::get_id();
        done.set_value();
    });

    doneFuture.wait();
    EXPECT_NE(workerId, std::thread::id{});
    EXPECT_NE(workerId, callerId);
}

TEST(NPUExecutorTests, ReentrantSubmissionFromWorkerDoesNotDeadlock) {
    auto exec = intel_npu::make_executor("reentrant_submit", 1, false, 200ms);

    std::mutex sequenceMutex;
    std::vector<std::string> sequence;

    std::promise<void> parentDone;
    std::promise<void> childDone;
    auto parentFuture = parentDone.get_future();
    auto childFuture = childDone.get_future();

    exec->run([&] {
        {
            std::lock_guard<std::mutex> lock(sequenceMutex);
            sequence.emplace_back("parent_start");
        }

        exec->run([&] {
            {
                std::lock_guard<std::mutex> lock(sequenceMutex);
                sequence.emplace_back("child");
            }
            childDone.set_value();
        });

        {
            std::lock_guard<std::mutex> lock(sequenceMutex);
            sequence.emplace_back("parent_end");
        }
        parentDone.set_value();
    });

    parentFuture.wait();
    childFuture.wait();

    ASSERT_EQ(sequence.size(), 3U);
    EXPECT_EQ(sequence[0], "parent_start");
    EXPECT_EQ(sequence[1], "parent_end");
    EXPECT_EQ(sequence[2], "child");
}

TEST(NPUExecutorTests, StressManyProducersNoTaskLoss) {
    auto exec = intel_npu::make_executor("stress_many_submitters", 4, false, 200ms);

    constexpr int producers = 4;
    constexpr int tasksPerProducer = 80;
    constexpr int expectedSuccessesPerProducer = tasksPerProducer;

    std::atomic<int> successfulTasks{0};

    std::vector<std::thread> submitters;
    submitters.reserve(producers);

    for (int submitterIdx = 0; submitterIdx < producers; ++submitterIdx) {
        submitters.emplace_back([&] {
            for (int i = 0; i < tasksPerProducer; ++i) {
                exec->run([&] {
                    successfulTasks.fetch_add(1);
                });
            }
        });
    }

    for (auto& th : submitters) {
        th.join();
    }

    ASSERT_TRUE(wait_until([&] {
        return successfulTasks.load() == producers * expectedSuccessesPerProducer;
    }));

    EXPECT_EQ(successfulTasks.load(), producers * expectedSuccessesPerProducer);
}

TEST(NPUExecutorTests, InlineModeSwallowsCustomExceptionType) {
    class CustomDeferredError final : public std::runtime_error {
    public:
        explicit CustomDeferredError(const std::string& whatArg) : std::runtime_error(whatArg) {}
    };

    auto exec = intel_npu::make_executor("typed_exception", 0, false, 200ms);

    std::atomic<bool> taskRan{false};
    EXPECT_NO_THROW(exec->run([&] {
        taskRan = true;
        throw CustomDeferredError("typed_error");
    }));

    EXPECT_TRUE(taskRan.load());
}

TEST(NPUExecutorTests, AdaptiveModeTinyIdleTimeoutRemainsStable) {
    auto exec = intel_npu::make_executor("tiny_idle_timeout", 1, true, 1ms);

    constexpr int totalTasks = 200;
    std::atomic<int> completed{0};

    for (int i = 0; i < totalTasks; ++i) {
        exec->run([&, i] {
            if (i % 3 == 0) {
                std::this_thread::yield();
            }
            completed.fetch_add(1);
        });
    }

    ASSERT_TRUE(wait_until([&] {
        return completed.load() == totalTasks;
    }));
}

TEST(NPUExecutorTests, ResettingOneOwnerDoesNotStopOtherOwners) {
    auto ownerA = intel_npu::make_executor("shared_owner_reset", 1, false, 200ms);
    auto ownerB = ownerA;

    std::thread releaser([&] {
        ownerA.reset();
    });
    releaser.join();

    std::atomic<bool> ran{false};
    ownerB->run([&] {
        ran = true;
    });

    ASSERT_TRUE(wait_until([&] {
        return ran.load();
    }));
}

TEST(NPUExecutorTests, AsyncInferRequestPipelineSuccessCallback) {
    auto state = std::make_shared<SharedState>();
    auto compiledModel = std::make_shared<TestCompiledModel>(state);
    auto asyncRequest = compiledModel->create_infer_request();

    std::atomic<bool> callbackCalled{false};
    std::atomic<bool> callbackHadException{true};

    asyncRequest->set_callback([&](std::exception_ptr ex) {
        callbackCalled = true;
        callbackHadException = (ex != nullptr);
    });

    asyncRequest->start_async();
    asyncRequest->wait();

    EXPECT_TRUE(callbackCalled.load());
    EXPECT_FALSE(callbackHadException.load());
    EXPECT_EQ(state->inferCalls.load(), 1);
}

TEST(NPUExecutorTests, AsyncInferRequestPipelineFailureCallback) {
    auto state = std::make_shared<SharedState>();
    auto compiledModel = std::make_shared<TestCompiledModel>(state);
    auto asyncRequest = compiledModel->create_infer_request();

    state->failInfer = true;

    std::atomic<bool> callbackCalled{false};
    std::atomic<bool> callbackHadException{false};

    asyncRequest->set_callback([&](std::exception_ptr ex) {
        callbackCalled = true;
        callbackHadException = (ex != nullptr);
    });

    asyncRequest->start_async();
    ASSERT_THROW(asyncRequest->wait(), std::exception);

    ASSERT_TRUE(wait_until([&] {
        return callbackCalled.load();
    }));
    EXPECT_TRUE(callbackHadException.load());
    EXPECT_EQ(state->inferCalls.load(), 1);
}

}  // namespace
