// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/cpu_streams_executor.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace {

class CPUStreamsExecutorErrorTest : public testing::TestWithParam<std::tuple<int, int>> {
protected:
    std::unique_ptr<ov::threading::CPUStreamsExecutor> make_executor() const {
        return std::make_unique<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"ExceptionRegression",
                                                    std::get<0>(GetParam()),
                                                    std::get<1>(GetParam())});
    }
};

TEST_P(CPUStreamsExecutorErrorTest, PropagatesWithoutReplayingOrRetainingFailedTask) {
    auto executor = make_executor();
    int failed = 0;
    int recovered = 0;
    auto capture = std::make_shared<int>(1);
    std::weak_ptr<int> weak_capture = capture;

    EXPECT_THROW(executor->execute([&, capture] {
        ++failed;
        throw std::runtime_error("executor sentinel");
    }),
                 std::runtime_error);
    capture.reset();
    EXPECT_TRUE(weak_capture.expired());
    EXPECT_NO_THROW(executor->execute([&] {
        ++recovered;
    }));
    EXPECT_EQ(failed, 1);
    EXPECT_EQ(recovered, 1);
}

TEST_P(CPUStreamsExecutorErrorTest, DrainsRecursiveTasksInOrderAndRethrowsFirstError) {
    auto executor = make_executor();
    std::vector<int> order;

    try {
        executor->execute([&] {
            order.push_back(1);
            executor->execute([&] {
                order.push_back(3);
                executor->execute([&] {
                    order.push_back(5);
                });
                throw std::logic_error("child error");
            });
            executor->execute([&] {
                order.push_back(4);
            });
            order.push_back(2);
            throw std::runtime_error("parent error");
        });
        ADD_FAILURE() << "The synchronous caller did not receive the first error";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "parent error");
    } catch (...) {
        ADD_FAILURE() << "The synchronous caller received a later error";
    }
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3, 4, 5}));
    EXPECT_NO_THROW(executor->execute([&] {
        order.push_back(6);
    }));
    EXPECT_EQ(order, (std::vector<int>{1, 2, 3, 4, 5, 6}));
}

TEST_P(CPUStreamsExecutorErrorTest, IndependentCallersRecoverAfterErrors) {
    auto executor = make_executor();
    std::atomic<int> failed{0};
    std::atomic<int> reported{0};
    std::atomic<int> recovered{0};
    std::vector<std::thread> callers;
    for (int index = 0; index < 3; ++index) {
        callers.emplace_back([&] {
            try {
                executor->execute([&] {
                    ++failed;
                    throw std::runtime_error("caller error");
                });
            } catch (const std::runtime_error&) {
                ++reported;
            }
            executor->execute([&] {
                ++recovered;
            });
        });
    }
    for (auto& caller : callers) {
        caller.join();
    }
    EXPECT_EQ(failed.load(), 3);
    EXPECT_EQ(reported.load(), 3);
    EXPECT_EQ(recovered.load(), 3);
}

TEST_P(CPUStreamsExecutorErrorTest, RunPreservesSynchronousAndPackagedAsynchronousBehavior) {
    auto executor = make_executor();
    if (std::get<0>(GetParam()) == 0) {
        EXPECT_THROW(executor->run([] {
            throw std::runtime_error("run sentinel");
        }),
                     std::runtime_error);
        int recovered = 0;
        EXPECT_NO_THROW(executor->run([&] {
            ++recovered;
        }));
        EXPECT_EQ(recovered, 1);
    } else {
        auto failed = std::make_shared<std::packaged_task<void()>>([] {
            throw std::runtime_error("worker sentinel");
        });
        auto failed_result = failed->get_future();
        executor->run([failed] {
            (*failed)();
        });
        ASSERT_EQ(failed_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_THROW(failed_result.get(), std::runtime_error);

        auto recovered = std::make_shared<std::packaged_task<int()>>([] {
            return 42;
        });
        auto recovered_result = recovered->get_future();
        executor->run([recovered] {
            (*recovered)();
        });
        ASSERT_EQ(recovered_result.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_EQ(recovered_result.get(), 42);
    }
}

INSTANTIATE_TEST_SUITE_P(StreamsAndThreads,
                         CPUStreamsExecutorErrorTest,
                         testing::Combine(testing::Values(0, 1, 2), testing::Values(1, 2, 4)));

}  // namespace
