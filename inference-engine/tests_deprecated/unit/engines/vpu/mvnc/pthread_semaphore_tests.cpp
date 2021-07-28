// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _WIN32

#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <thread>
#include <atomic>
#include <limits.h>

#include "pthread_semaphore.h"

class PThreadSemaphoreTest : public ::testing::Test {
 public:
    pthread_sem_t  sem = 0;
};

class PThreadBinSemaphoreTest : public ::testing::TestWithParam<int>{
 public:
    pthread_sem_t  bin_sem = 0;
    void SetUp() override {
        ASSERT_EQ(0, pthread_sem_init(&bin_sem, 0, 1));
        ASSERT_NE(0, bin_sem);
        ASSERT_EQ(errno, 0);
    }

    void TearDown() override {
        if (0 != bin_sem) {
            ASSERT_EQ(0, pthread_sem_destroy(&bin_sem));
            ASSERT_EQ(0, bin_sem);
            ASSERT_EQ(errno, 0);
        }
    }
    /**
     * @brief special version of post that for timeout == -1 issue broadcast, and for other timeout values issues broadcast
     * @param timeout_override
     * @return
     */
    int invoke_post() {
        float timeout = GetParam();
        if (timeout == -1) {
            return pthread_sem_post_broadcast(&bin_sem);
        }
        return pthread_sem_post(&bin_sem);
    }

    int invoke_wait(float timeout_override = 0) {
        // infinite wait - lets call pthread_wait
        float timeout = GetParam();
        if (timeout_override != 0) {
            timeout = timeout_override;
        }
        if (timeout == -1) {
            return pthread_sem_wait(&bin_sem);
        }
        timespec spec = {};

        if (clock_gettime(CLOCK_REALTIME, &spec) == -1) {
            std::cerr << "clock_gettime";
        }

        auto newNsec = static_cast<long long>(spec.tv_nsec + timeout * 1000000000LL);
        spec.tv_sec   += newNsec / 1000000000L;
        spec.tv_nsec  =  newNsec % 1000000000L;

        return pthread_sem_timedwait(&bin_sem, &spec);
    }
};


TEST_F(PThreadSemaphoreTest, CanNotInitSemaWithZeroPointer) {
    ASSERT_EQ(-1, pthread_sem_init(nullptr, 0, 1));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, CanNotInitSemaWithBigCounter) {
    ASSERT_EQ(-1, pthread_sem_init(&sem, 0, static_cast<uint32_t>(std::numeric_limits<int>::max())+ 1));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, CanInitSemaWithMaxCounter) {
    ASSERT_EQ(0, pthread_sem_init(&sem, 0, SEM_VALUE_MAX));
    ASSERT_EQ(errno, 0);

    ASSERT_EQ(0, pthread_sem_destroy(&sem));
    ASSERT_EQ(errno, 0);
}

#ifdef ANDROID
TEST_F(PThreadSemaphoreTest, DISABLED_CanNotPostSemaWithMaxCounter) {
#else
TEST_F(PThreadSemaphoreTest, CanNotPostSemaWithMaxCounter) {
#endif
    ASSERT_EQ(0, pthread_sem_init(&sem, 0, SEM_VALUE_MAX));
    ASSERT_EQ(errno, 0);

    ASSERT_EQ(-1, pthread_sem_post(&sem));
    ASSERT_EQ(EOVERFLOW, errno);

    ASSERT_EQ(0, pthread_sem_destroy(&sem));
    ASSERT_EQ(errno, 0);
}

TEST_F(PThreadSemaphoreTest, CanNotInitSemaWithSystemNonZero) {
    ASSERT_EQ(-1, pthread_sem_init(&sem, 1, 1));
    ASSERT_EQ(errno, ENOSYS);
}

TEST_F(PThreadSemaphoreTest, DestroyOfNonInit) {
    ASSERT_EQ(-1, pthread_sem_destroy(nullptr));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, DestroyOfNonAllocated) {
    ASSERT_EQ(-1, pthread_sem_destroy(&sem));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, PostOfNonInit) {
    ASSERT_EQ(-1, pthread_sem_post(nullptr));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, PostOfNonAllocated) {
    ASSERT_EQ(-1, pthread_sem_post(&sem));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, WaitOfNonInit) {
    ASSERT_EQ(-1, pthread_sem_wait(nullptr));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, WaitOfNonAllocated) {
    ASSERT_EQ(-1, pthread_sem_wait(&sem));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, TimedWaitOfNonInit) {
    ASSERT_EQ(-1, pthread_sem_timedwait(nullptr, nullptr));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, TimedWaitOfNonAllocated) {
    ASSERT_EQ(-1, pthread_sem_timedwait(&sem, nullptr));
    ASSERT_EQ(errno, EINVAL);
}

TEST_F(PThreadSemaphoreTest, TimedWaitOfNullptrTimeval) {
    ASSERT_EQ(0, pthread_sem_init(&sem, 0, 1));

    ASSERT_EQ(-1, pthread_sem_timedwait(&sem, nullptr));
    ASSERT_EQ(errno, EINVAL);

    ASSERT_EQ(0, pthread_sem_destroy(&sem));
    ASSERT_EQ(errno, 0);
}

TEST_F(PThreadSemaphoreTest, TimedWaitOfNegativeInterval) {
    ASSERT_EQ(0, pthread_sem_init(&sem, 0, 1));

    timespec timeout = {};
    timeout.tv_sec = -1;
    ASSERT_EQ(-1, pthread_sem_timedwait(&sem, &timeout));
    ASSERT_EQ(errno, EINVAL);

    timeout.tv_sec = 0;
    timeout.tv_nsec = -1;
    ASSERT_EQ(-1, pthread_sem_timedwait(&sem, &timeout));
    ASSERT_EQ(errno, EINVAL);

    ASSERT_EQ(0, pthread_sem_destroy(&sem));
    ASSERT_EQ(errno, 0);
}

TEST_P(PThreadBinSemaphoreTest, CanInitSema) {
    SUCCEED();
}

TEST_P(PThreadBinSemaphoreTest, DoubleDestroyOfSema) {
    ASSERT_EQ(0, pthread_sem_destroy(&bin_sem));
    ASSERT_EQ(0, bin_sem);
    ASSERT_EQ(errno, 0);

    ASSERT_EQ(-1, pthread_sem_destroy(&bin_sem));
    ASSERT_EQ(0, bin_sem);
    ASSERT_EQ(errno, EINVAL);
}

TEST_P(PThreadBinSemaphoreTest, CanAcquireSemaOnce) {
    // non blocked
    ASSERT_EQ(0, invoke_wait());
    ASSERT_EQ(0, pthread_sem_post(&bin_sem));
}

TEST_P(PThreadBinSemaphoreTest, WillBlockWhenAcquiringSemaTwice) {
    // non blocked
    ASSERT_EQ(0, invoke_wait());
    bool secondLock = true;

    std::thread th ([&]() {
        // this is enough to delay thread start and block on second acquire of sema by main thread
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ASSERT_TRUE(secondLock);
        ASSERT_EQ(0, pthread_sem_post(&bin_sem));
    });

    ASSERT_EQ(0, invoke_wait());
    // marked as non blocked
    secondLock = false;

    th.join();
}

TEST_P(PThreadBinSemaphoreTest, WillBlockWhenAcquiringSemaTwice2) {
    // non blocked
    ASSERT_EQ(0, invoke_wait());  // counter = 0
    bool secondLock = true;

    std::thread th([&]() {
        // this is enough to delay thread start and block on second acquire of sema by main thread
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ASSERT_TRUE(secondLock);
        secondLock = false;
        // unblock both
        ASSERT_EQ(0, pthread_sem_post(&bin_sem));  // counter = 0
        // block until first tread helps us
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ASSERT_EQ(0, invoke_wait());  // counter = -1 - block
        ASSERT_TRUE(secondLock);
    });

    ASSERT_EQ(0, pthread_sem_wait(&bin_sem));  // counter = -1 - block

    // once condition variable signaled lets other thread to acquire it
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ASSERT_FALSE(secondLock);

    // marked as non blocked
    secondLock = true;

    // unblock second thread
    ASSERT_EQ(0, pthread_sem_post(&bin_sem));  // counter = 0 - block

    th.join();
}

TEST_P(PThreadBinSemaphoreTest, DestroyAcquiredSemaResultedInError) {
    ASSERT_EQ(0, invoke_wait());

    // semaphore deleted - since not blocked, even if counter is 0
    ASSERT_EQ(0, pthread_sem_destroy(&bin_sem));
    ASSERT_EQ(0, bin_sem);

    // reinit sema
    SetUp();
    ASSERT_EQ(0, invoke_wait());  // --counter = 1
    std::thread th([&]() {
        // enough time for main thread to became blocked
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ASSERT_EQ(-1, pthread_sem_destroy(&bin_sem));
        ASSERT_EQ(EBUSY, errno);
        ASSERT_NE(0, bin_sem);
        ASSERT_EQ(0, pthread_sem_post(&bin_sem));
    });
    ASSERT_EQ(0, invoke_wait());  // --counter = 0 - blocked

    th.join();
}

TEST_P(PThreadBinSemaphoreTest, TimedWaitFinallysucceed) {
    if (GetParam() < 0) GTEST_SKIP();

    ASSERT_EQ(0, invoke_wait());

    std::thread th([&]() {
        // enough time for main thread to became blocked
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ASSERT_EQ(0, pthread_sem_post(&bin_sem));
    });
    int result = 0;
    ASSERT_EQ(-1, result = invoke_wait(0.1));  // right now sema gets occupied and resulted of a timeout
    ASSERT_EQ(ETIMEDOUT, errno);
    int i = 0;
    for (i = 0; i < 100; i++) {
        result = invoke_wait(0.1);
        if (0 == result) {
            break;
        }
        ASSERT_EQ(ETIMEDOUT, errno) << "actual errno value=" << result;

    }
    // so 100 x 100 ms timeout should be enough
    ASSERT_EQ(result, 0);

    th.join();
}

TEST_P(PThreadBinSemaphoreTest, TimedWaitDidntIncreaseCounter) {
    if (GetParam() < 0) GTEST_SKIP();

    ASSERT_EQ(0, invoke_wait());  // counter = 0
    bool locked = true;

    std::thread th([&]() {
        // enough time for main thread to became blocked
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ASSERT_TRUE(locked);
        locked = false;
        ASSERT_EQ(0, pthread_sem_post(&bin_sem));
    });
    ASSERT_EQ(-1, invoke_wait(0.1));  // right now sema gets occupied and resulted of a timeout
    ASSERT_EQ(ETIMEDOUT, errno);

    // single post still is enough
    ASSERT_EQ(0, invoke_wait(-1));  // blocking here
    ASSERT_FALSE(locked);

    th.join();
}

TEST_P(PThreadBinSemaphoreTest, PostWakeUpOnlyOneWaiterOfMany) {
    // non blocked
    ASSERT_EQ(0, invoke_wait()); // counter = 0

    std::atomic<int> num_threads_woke(0);

    std::thread th1([&]() {
        ASSERT_EQ(0, invoke_wait());
        num_threads_woke++;
    });

    std::thread th2([&]() {
        ASSERT_EQ(0, invoke_wait());
        num_threads_woke++;
    });

    std::thread th3([&]() {
        ASSERT_EQ(0, invoke_wait());
        num_threads_woke++;
    });

    // all 3 threads gets locked
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // unblock only 1 waiter
    ASSERT_EQ(0, invoke_post());
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ASSERT_EQ(1, num_threads_woke);

    // unblock second waiter
    ASSERT_EQ(0, invoke_post());
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ASSERT_EQ(2, num_threads_woke);

    // unblock last waiter
    ASSERT_EQ(0, invoke_post());
    std::this_thread::sleep_for(std::chrono::seconds(1));
    ASSERT_EQ(3, num_threads_woke);

    th1.join();
    th2.join();
    th3.join();
}

INSTANTIATE_TEST_SUITE_P(
    PThreadParametrizedTests,
    PThreadBinSemaphoreTest,
    ::testing::Values(
        -1, 100  // 10 seconds are quite long enough for time being
    ));

#endif  // _WIN32
