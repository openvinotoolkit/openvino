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

bool wait_until(const std::function<bool()>& pred, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(2ms);
    }
    return pred();
}

TEST(NPUExecutorTests, InlineModeRunsOnCallerThread) {
    auto exec = intel_npu::make_executor("inline", 0, false, 10ms);

    std::thread::id taskThread{};
    exec->run([&] {
        taskThread = std::this_thread::get_id();
    });

    EXPECT_EQ(taskThread, std::this_thread::get_id());
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

    ASSERT_TRUE(wait_until(
        [&] {
            std::lock_guard<std::mutex> lock(m);
            return started == 2;
        },
        200ms));

    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();
}

TEST(NPUExecutorTests, ThrowsOnNextRunForSameSubmitter) {
    auto exec = intel_npu::make_executor("throw_same", 1, false, 200ms);

    exec->run([] {
        throw std::runtime_error("boom_same_submitter");
    });

    ASSERT_TRUE(wait_until(
        [&] {
            try {
                exec->run([] {});
                return false;
            } catch (const std::runtime_error& ex) {
                return std::string(ex.what()) == "boom_same_submitter";
            }
        },
        500ms));
}

TEST(NPUExecutorTests, DoesNotThrowForDifferentSubmitterThread) {
    auto exec = intel_npu::make_executor("throw_other", 1, false, 200ms);

    exec->run([] {
        throw std::runtime_error("boom_other_submitter");
    });

    std::atomic<bool> wrongThreadThrew{false};
    std::thread other([&] {
        for (int i = 0; i < 10; ++i) {
            try {
                exec->run([] {});
            } catch (...) {
                wrongThreadThrew = true;
                break;
            }
            std::this_thread::sleep_for(5ms);
        }
    });
    other.join();

    EXPECT_FALSE(wrongThreadThrew.load());

    ASSERT_TRUE(wait_until(
        [&] {
            try {
                exec->run([] {});
                return false;
            } catch (const std::runtime_error& ex) {
                return std::string(ex.what()) == "boom_other_submitter";
            }
        },
        500ms));
}

TEST(NPUExecutorTests, PreservesMultipleExceptionsForSameSubmitter) {
    auto exec = intel_npu::make_executor("throw_many", 1, false, 200ms);

    exec->run([] {
        throw std::runtime_error("boom_1");
    });
    exec->run([] {
        throw std::runtime_error("boom_2");
    });

    std::set<std::string> observed;
    ASSERT_TRUE(wait_until(
        [&] {
            try {
                exec->run([] {});
            } catch (const std::runtime_error& ex) {
                observed.insert(ex.what());
            }
            return observed.size() == 2;
        },
        800ms));

    EXPECT_EQ(observed.count("boom_1"), 1U);
    EXPECT_EQ(observed.count("boom_2"), 1U);
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

    ASSERT_TRUE(wait_until(
        [&] {
            std::lock_guard<std::mutex> lock(m);
            return started >= workers;
        },
        300ms));

    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();

    ASSERT_TRUE(wait_until(
        [&] {
            std::lock_guard<std::mutex> lock(m);
            return completed == tasksCount;
        },
        800ms));

    std::lock_guard<std::mutex> lock(m);
    EXPECT_LE(workerThreads.size(), static_cast<size_t>(workers));
}

TEST(NPUExecutorTests, RethrowsPendingExceptionsInFifoOrderForSameSubmitter) {
    auto exec = intel_npu::make_executor("throw_fifo", 1, false, 200ms);

    std::mutex m;
    std::condition_variable cv;
    int throwTasksStarted = 0;

    auto enqueueThrow = [&](const char* msg) {
        exec->run([&, msg] {
            {
                std::lock_guard<std::mutex> lock(m);
                ++throwTasksStarted;
            }
            cv.notify_all();
            throw std::runtime_error(msg);
        });
    };

    enqueueThrow("fifo_1");
    enqueueThrow("fifo_2");

    ASSERT_TRUE(wait_until(
        [&] {
            std::lock_guard<std::mutex> lock(m);
            return throwTasksStarted == 2;
        },
        500ms));

    std::vector<std::string> observed;
    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(wait_until(
            [&] {
                try {
                    exec->run([] {});
                    return false;
                } catch (const std::runtime_error& ex) {
                    observed.emplace_back(ex.what());
                    return true;
                }
            },
            500ms));
    }

    ASSERT_EQ(observed.size(), 2U);
    EXPECT_EQ(observed[0], "fifo_1");
    EXPECT_EQ(observed[1], "fifo_2");
}

TEST(NPUExecutorTests, EachSubmitterGetsOnlyOwnPendingException) {
    auto exec = intel_npu::make_executor("throw_attr", 1, false, 200ms);

    auto submitter = [&](const std::string& expected, const std::string& other) {
        exec->run([&] {
            throw std::runtime_error(expected);
        });

        // Drain until this submitter receives its own exception.
        return wait_until(
            [&] {
                try {
                    exec->run([] {});
                    return false;
                } catch (const std::runtime_error& ex) {
                    EXPECT_NE(std::string(ex.what()), other);
                    return std::string(ex.what()) == expected;
                }
            },
            800ms);
    };

    std::promise<bool> pA;
    std::promise<bool> pB;

    std::thread tA([&] {
        pA.set_value(submitter("A_fail", "B_fail"));
    });
    std::thread tB([&] {
        pB.set_value(submitter("B_fail", "A_fail"));
    });

    tA.join();
    tB.join();

    EXPECT_TRUE(pA.get_future().get());
    EXPECT_TRUE(pB.get_future().get());
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

        ASSERT_TRUE(wait_until(
            [&] {
                std::lock_guard<std::mutex> lock(m);
                return started == 2;
            },
            300ms));

        {
            std::lock_guard<std::mutex> lock(m);
            release = true;
        }
        cv.notify_all();

        ASSERT_TRUE(wait_until(
            [&] {
                std::lock_guard<std::mutex> lock(m);
                return completed == 2;
            },
            300ms));

        // Give adaptive workers time to shrink back toward baseline.
        std::this_thread::sleep_for(50ms);
    }
}

TEST(NPUExecutorTests, PendingExceptionIsRethrownBeforeNewTaskIsQueued) {
    auto exec = intel_npu::make_executor("rethrow_before_enqueue", 1, false, 200ms);

    std::atomic<int> throwTasksExecuted{0};
    exec->run([&] {
        throwTasksExecuted.fetch_add(1);
        throw std::runtime_error("pending_before_enqueue_1");
    });
    exec->run([&] {
        throwTasksExecuted.fetch_add(1);
        throw std::runtime_error("pending_before_enqueue_2");
    });

    ASSERT_TRUE(wait_until(
        [&] {
            return throwTasksExecuted.load() == 2;
        },
        500ms));

    // Consume the first pending exception using a side-effect-free run().
    ASSERT_TRUE(wait_until(
        [&] {
            try {
                exec->run([] {});
                return false;
            } catch (const std::runtime_error& ex) {
                return std::string(ex.what()) == "pending_before_enqueue_1";
            }
        },
        500ms));

    std::atomic<bool> newTaskRan{false};

    // The second pending exception must be rethrown before enqueueing a new task.
    ASSERT_THROW(exec->run([&] {
        newTaskRan = true;
    }),
                 std::runtime_error);

    EXPECT_FALSE(newTaskRan.load());

    // Executor must still be usable after pending exception is consumed.
    exec->run([&] {
        newTaskRan = true;
    });
    ASSERT_TRUE(wait_until(
        [&] {
            return newTaskRan.load();
        },
        200ms));
}

TEST(NPUExecutorTests, MultiThreadedSubmittersGetOwnExceptionsAndSuccessThreadsStayClean) {
    auto exec = intel_npu::make_executor("multi_submitters", 2, false, 200ms);

    constexpr int throwSubmitters = 3;
    constexpr int failuresPerSubmitter = 3;
    constexpr int successSubmitterTasks = 20;

    std::atomic<bool> wrongException{false};
    std::atomic<bool> successSubmitterThrew{false};
    std::atomic<int> successTasksExecuted{0};

    std::vector<std::thread> workers;
    workers.reserve(throwSubmitters + 1);

    for (int submitterIdx = 0; submitterIdx < throwSubmitters; ++submitterIdx) {
        workers.emplace_back([&, submitterIdx] {
            const std::string token = "submitter_" + std::to_string(submitterIdx);

            for (int i = 0; i < failuresPerSubmitter; ++i) {
                exec->run([token] {
                    throw std::runtime_error(token);
                });
            }

            int received = 0;
            const auto deadline = std::chrono::steady_clock::now() + 2s;
            while (received < failuresPerSubmitter && std::chrono::steady_clock::now() < deadline) {
                try {
                    exec->run([] {});
                } catch (const std::runtime_error& ex) {
                    if (std::string(ex.what()) != token) {
                        wrongException = true;
                    }
                    ++received;
                }
                std::this_thread::sleep_for(1ms);
            }

            if (received != failuresPerSubmitter) {
                wrongException = true;
            }
        });
    }

    workers.emplace_back([&] {
        for (int i = 0; i < successSubmitterTasks; ++i) {
            try {
                exec->run([&] {
                    successTasksExecuted.fetch_add(1);
                });
            } catch (...) {
                successSubmitterThrew = true;
            }
            std::this_thread::sleep_for(1ms);
        }
    });

    for (auto& th : workers) {
        th.join();
    }

    EXPECT_FALSE(wrongException.load());
    EXPECT_FALSE(successSubmitterThrew.load());
    EXPECT_EQ(successTasksExecuted.load(), successSubmitterTasks);
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

    ASSERT_TRUE(wait_until(
        [&] {
            std::lock_guard<std::mutex> lock(m);
            return started;
        },
        300ms));

    std::thread destroyer([&] {
        exec.reset();
        destructorReturned = true;
    });

    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(destructorReturned.load());

    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();

    destroyer.join();
    EXPECT_TRUE(finished.load());
    EXPECT_TRUE(destructorReturned.load());
}

TEST(NPUExecutorTests, InterleavedSubmittersKeepOwnExceptionOrder) {
    auto exec = intel_npu::make_executor("interleaved_attr_fifo", 2, false, 200ms);

    std::promise<void> submitDoneA;
    std::promise<void> submitDoneB;
    auto submitReadyA = submitDoneA.get_future();
    auto submitReadyB = submitDoneB.get_future();

    std::promise<std::vector<std::string>> seenA;
    std::promise<std::vector<std::string>> seenB;

    std::thread submitterA([&] {
        exec->run([] {
            throw std::runtime_error("A_1");
        });
        exec->run([] {
            throw std::runtime_error("A_2");
        });
        submitDoneA.set_value();

        std::vector<std::string> local;
        while (local.size() < 2) {
            try {
                exec->run([] {});
            } catch (const std::runtime_error& ex) {
                if (std::string(ex.what()).rfind("A_", 0) == 0) {
                    local.push_back(ex.what());
                }
            }
            std::this_thread::sleep_for(1ms);
        }
        seenA.set_value(std::move(local));
    });

    std::thread submitterB([&] {
        exec->run([] {
            throw std::runtime_error("B_1");
        });
        exec->run([] {
            throw std::runtime_error("B_2");
        });
        submitDoneB.set_value();

        std::vector<std::string> local;
        while (local.size() < 2) {
            try {
                exec->run([] {});
            } catch (const std::runtime_error& ex) {
                if (std::string(ex.what()).rfind("B_", 0) == 0) {
                    local.push_back(ex.what());
                }
            }
            std::this_thread::sleep_for(1ms);
        }
        seenB.set_value(std::move(local));
    });

    submitReadyA.wait();
    submitReadyB.wait();

    submitterA.join();
    submitterB.join();

    const auto observedA = seenA.get_future().get();
    const auto observedB = seenB.get_future().get();

    ASSERT_EQ(observedA.size(), 2U);
    ASSERT_EQ(observedB.size(), 2U);

    // With multiple workers, completion order is not deterministic.
    // Validate ownership and completeness; FIFO is covered by single-worker tests.
    std::multiset<std::string> expectedA{"A_1", "A_2"};
    std::multiset<std::string> expectedB{"B_1", "B_2"};
    std::multiset<std::string> actualA(observedA.begin(), observedA.end());
    std::multiset<std::string> actualB(observedB.begin(), observedB.end());

    EXPECT_EQ(actualA, expectedA);
    EXPECT_EQ(actualB, expectedB);
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

    ASSERT_EQ(doneFuture.wait_for(500ms), std::future_status::ready);
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

    ASSERT_EQ(parentFuture.wait_for(500ms), std::future_status::ready);
    ASSERT_EQ(childFuture.wait_for(500ms), std::future_status::ready);

    ASSERT_EQ(sequence.size(), 3U);
    EXPECT_EQ(sequence[0], "parent_start");
    EXPECT_EQ(sequence[1], "parent_end");
    EXPECT_EQ(sequence[2], "child");
}

TEST(NPUExecutorTests, StressManyProducersNoTaskOrExceptionLoss) {
    auto exec = intel_npu::make_executor("stress_many_submitters", 4, false, 200ms);

    constexpr int producers = 4;
    constexpr int tasksPerProducer = 80;
    constexpr int failEvery = 5;
    constexpr int expectedFailuresPerProducer = tasksPerProducer / failEvery;
    constexpr int expectedSuccessesPerProducer = tasksPerProducer - expectedFailuresPerProducer;

    std::atomic<int> successfulTasks{0};
    std::atomic<bool> wrongExceptionOwner{false};
    std::atomic<bool> missedExceptions{false};

    std::vector<std::thread> submitters;
    submitters.reserve(producers);

    for (int submitterIdx = 0; submitterIdx < producers; ++submitterIdx) {
        submitters.emplace_back([&, submitterIdx] {
            const std::string token = "stress_" + std::to_string(submitterIdx) + "_";

            for (int i = 0; i < tasksPerProducer; ++i) {
                exec->run([&, i, token] {
                    if ((i + 1) % failEvery == 0) {
                        throw std::runtime_error(token + std::to_string(i));
                    }
                    successfulTasks.fetch_add(1);
                });
            }

            int seenFailures = 0;
            const auto deadline = std::chrono::steady_clock::now() + 3s;
            while (seenFailures < expectedFailuresPerProducer && std::chrono::steady_clock::now() < deadline) {
                try {
                    exec->run([] {});
                } catch (const std::runtime_error& ex) {
                    const std::string msg = ex.what();
                    if (msg.rfind(token, 0) != 0) {
                        wrongExceptionOwner = true;
                    }
                    ++seenFailures;
                }
                std::this_thread::sleep_for(1ms);
            }

            if (seenFailures != expectedFailuresPerProducer) {
                missedExceptions = true;
            }
        });
    }

    for (auto& th : submitters) {
        th.join();
    }

    ASSERT_TRUE(wait_until(
        [&] {
            return successfulTasks.load() == producers * expectedSuccessesPerProducer;
        },
        2s));

    EXPECT_FALSE(wrongExceptionOwner.load());
    EXPECT_FALSE(missedExceptions.load());
    EXPECT_EQ(successfulTasks.load(), producers * expectedSuccessesPerProducer);
}

TEST(NPUExecutorTests, DeferredRethrowPreservesCustomExceptionType) {
    class CustomDeferredError final : public std::runtime_error {
    public:
        explicit CustomDeferredError(const std::string& whatArg) : std::runtime_error(whatArg) {}
    };

    auto exec = intel_npu::make_executor("typed_exception", 1, false, 200ms);

    exec->run([] {
        throw CustomDeferredError("typed_error");
    });

    bool caughtTyped = false;
    ASSERT_TRUE(wait_until(
        [&] {
            try {
                exec->run([] {});
                return false;
            } catch (const CustomDeferredError& ex) {
                caughtTyped = true;
                return std::string(ex.what()) == "typed_error";
            } catch (...) {
                return false;
            }
        },
        500ms));

    EXPECT_TRUE(caughtTyped);
}

TEST(NPUExecutorTests, AdaptiveModeTinyIdleTimeoutRemainsStable) {
    auto exec = intel_npu::make_executor("tiny_idle_timeout", 1, true, 1ms);

    constexpr int totalTasks = 200;
    std::atomic<int> completed{0};

    for (int i = 0; i < totalTasks; ++i) {
        exec->run([&, i] {
            if (i % 3 == 0) {
                std::this_thread::sleep_for(1ms);
            }
            completed.fetch_add(1);
        });
    }

    ASSERT_TRUE(wait_until(
        [&] {
            return completed.load() == totalTasks;
        },
        3s));
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

    ASSERT_TRUE(wait_until(
        [&] {
            return ran.load();
        },
        300ms));
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

    ASSERT_TRUE(wait_until(
        [&] {
            return callbackCalled.load();
        },
        200ms));
    EXPECT_TRUE(callbackHadException.load());
    EXPECT_EQ(state->inferCalls.load(), 1);
}

}  // namespace
