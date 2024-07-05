// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <future>
#include <thread>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"

using namespace ::testing;
using namespace std;
using namespace ov;
using namespace threading;

static constexpr const auto MAX_NUMBER_OF_TASKS_IN_QUEUE = 10;

using Future = std::future<void>;

class TaskExecutorTests : public ::testing::TestWithParam<std::function<ITaskExecutor::Ptr()>> {};

TEST_P(TaskExecutorTests, canCreateTaskExecutor) {
    auto makeExecutor = GetParam();
    EXPECT_NO_THROW(makeExecutor());
}

template <typename E, typename F>
static std::future<void> async(E& executor, F&& f) {
    auto p = std::make_shared<std::packaged_task<void()>>(f);
    auto future = p->get_future();
    executor->run([p] {
        (*p)();
    });
    return future;
}

TEST_P(TaskExecutorTests, canRunCustomFunction) {
    auto taskExecutor = GetParam()();
    int i = 0;
    auto f = async(taskExecutor, [&i] {
        i++;
    });
    f.wait();
    OV_ASSERT_NO_THROW(f.get());
}

TEST_P(TaskExecutorTests, canRun2FunctionsOneByOne) {
    auto taskExecutor = GetParam()();
    std::mutex m;
    int i = 0;
    auto f1 = async(taskExecutor, [&]() {
        std::unique_lock<std::mutex> l{m};
        i += 1;
    });
    f1.wait();
    OV_ASSERT_NO_THROW(f1.get());
    auto f2 = async(taskExecutor, [&]() {
        std::unique_lock<std::mutex> l{m};
        i *= 2;
    });
    f2.wait();
    OV_ASSERT_NO_THROW(f2.get());

    ASSERT_EQ(i, 2);
}

TEST_P(TaskExecutorTests, canRun2FunctionsOneByOneWithoutWait) {
    auto taskExecutor = GetParam()();
    async(taskExecutor, [] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    });
    async(taskExecutor, [] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    });
}

TEST_P(TaskExecutorTests, canRunMultipleTasksWithExceptionInside) {
    auto taskExecutor = GetParam()();
    std::vector<std::future<void>> futures;

    for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE; i++) {
        futures.emplace_back(async(taskExecutor, [] {
            throw std::bad_alloc();
        }));
    }

    for (auto& f : futures) {
        f.wait();
        EXPECT_THROW(f.get(), std::bad_alloc);
    }
}

// TODO: Issue-11695
TEST_P(TaskExecutorTests, canRunMultipleTasksFromMultipleThreads) {
    auto taskExecutor = GetParam()();
    std::atomic_int sharedVar = {0};
    int THREAD_NUMBER = MAX_NUMBER_OF_TASKS_IN_QUEUE;
    int NUM_INTERNAL_ITERATIONS = 5000;
    std::vector<std::thread> threads;
    std::vector<Future> futures;
    for (int i = 0; i < THREAD_NUMBER; i++) {
        auto p = std::make_shared<std::packaged_task<void()>>([&] {
            for (int k = 0; k < NUM_INTERNAL_ITERATIONS; k++) {
                ++sharedVar;
            }
        });
        futures.emplace_back(p->get_future());
        auto task = [p] {
            (*p)();
        };
        threads.emplace_back([task, taskExecutor] {
            taskExecutor->run(std::move(task));
        });
    }

    for (auto&& f : futures)
        f.wait();
    for (auto&& f : futures)
        OV_ASSERT_NO_THROW(f.get());
    ASSERT_EQ(THREAD_NUMBER * NUM_INTERNAL_ITERATIONS, sharedVar);
    for (auto&& thread : threads)
        if (thread.joinable())
            thread.join();
}

TEST_P(TaskExecutorTests, executorNotReleasedUntilTasksAreDone) {
    std::mutex mutex_block_emulation;
    std::condition_variable cv_block_emulation;
    std::vector<Future> futures;
    bool isBlocked = true;
    std::atomic_int sharedVar = {0};
    {
        auto taskExecutor = GetParam()();
        for (int i = 0; i < MAX_NUMBER_OF_TASKS_IN_QUEUE; i++) {
            auto p = std::make_shared<std::packaged_task<void()>>([&] {
                // intentionally block task for launching tasks after calling dtor for TaskExecutor
                std::unique_lock<std::mutex> lock(mutex_block_emulation);
                cv_block_emulation.wait(lock, [&isBlocked] {
                    return isBlocked;
                });
                ++sharedVar;
            });
            futures.emplace_back(p->get_future());
            auto task = [p] {
                (*p)();
            };
            taskExecutor->run(std::move(task));
        }
    }
    // time to call dtor for taskExecutor and unlock tasks
    {
        std::lock_guard<std::mutex> lock{mutex_block_emulation};
        isBlocked = false;
    }
    for (auto& f : futures) {
        cv_block_emulation.notify_all();
        f.wait();
    }
    // all tasks should be called despite calling dtor for TaskExecutor
    ASSERT_EQ(MAX_NUMBER_OF_TASKS_IN_QUEUE, sharedVar);
}

class ASyncTaskExecutorTests : public TaskExecutorTests {};

// TODO: Issue-11695
TEST_P(ASyncTaskExecutorTests, startAsyncIsNotBlockedByAnotherTask) {
    std::mutex mutex_block_emulation;
    std::condition_variable cv_block_emulation;
    std::mutex mutex_task_started;
    std::condition_variable cv_task_started;
    bool isStarted = false;
    bool isBlocked = true;
    auto taskExecutor = GetParam()();

    async(taskExecutor, [&] {
        {
            std::lock_guard<std::mutex> lock(mutex_task_started);
            isStarted = true;
        }
        cv_task_started.notify_all();
        // intentionally block task for test purpose
        std::unique_lock<std::mutex> lock(mutex_block_emulation);
        cv_block_emulation.wait(lock, [&isBlocked] {
            return !isBlocked;
        });
    });

    async(taskExecutor, [&] {
        std::unique_lock<std::mutex> lock(mutex_task_started);
        cv_task_started.wait(lock, [&isStarted] {
            return isStarted;
        });
    });

    {
        std::lock_guard<std::mutex> lock(mutex_block_emulation);
        isBlocked = false;
    }
    cv_block_emulation.notify_all();
}

TEST_P(ASyncTaskExecutorTests, runAndWaitDoesNotOwnTasks) {
    std::shared_ptr<void> sharedCounter(this, [](ASyncTaskExecutorTests*) {});
    auto taskExecutor = GetParam()();
    std::atomic_int useCount = {0};
    std::vector<Task> tasks = {[sharedCounter, &useCount] {
        useCount = sharedCounter.use_count();
    }};
    sharedCounter.reset();
    taskExecutor->run_and_wait(tasks);
    ASSERT_EQ(1, useCount);
}

class StreamsExecutorConfigTest : public ::testing::Test {};

static auto Executors = ::testing::Values(
    [] {
        auto streams = get_number_of_cpu_cores();
        auto threads = parallel_get_max_threads();
        return std::make_shared<CPUStreamsExecutor>(
            IStreamsExecutor::Config{"TestCPUStreamsExecutor", streams, threads / streams});
    },
    [] {
        auto streams = get_number_of_logical_cpu_cores(true);
        auto threads = parallel_get_max_threads();
        return std::make_shared<CPUStreamsExecutor>(
            IStreamsExecutor::Config{"TestCPUStreamsExecutor", streams, threads / streams});
    },
    [] {
        auto streams = get_number_of_logical_cpu_cores(false);
        auto threads = parallel_get_max_threads();
        return std::make_shared<CPUStreamsExecutor>(
            IStreamsExecutor::Config{"TestCPUStreamsExecutor", streams, threads / streams});
    },
    [] {
        return std::make_shared<ImmediateExecutor>();
    });

INSTANTIATE_TEST_SUITE_P(TaskExecutorTests, TaskExecutorTests, Executors);

static auto AsyncExecutors = ::testing::Values(
    [] {
        auto streams = get_number_of_cpu_cores();
        auto threads = parallel_get_max_threads();
        return std::make_shared<CPUStreamsExecutor>(
            IStreamsExecutor::Config{"TestCPUStreamsExecutor", streams, threads / streams});
    },
    [] {
        auto streams = get_number_of_logical_cpu_cores(true);
        auto threads = parallel_get_max_threads();
        return std::make_shared<CPUStreamsExecutor>(
            IStreamsExecutor::Config{"TestCPUStreamsExecutor", streams, threads / streams});
    },
    [] {
        auto streams = get_number_of_logical_cpu_cores(false);
        auto threads = parallel_get_max_threads();
        return std::make_shared<CPUStreamsExecutor>(
            IStreamsExecutor::Config{"TestCPUStreamsExecutor", streams, threads / streams});
    });

INSTANTIATE_TEST_SUITE_P(ASyncTaskExecutorTests, ASyncTaskExecutorTests, AsyncExecutors);
