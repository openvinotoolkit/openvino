// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_guard.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace ov::test {

TEST(CacheGuardEntryTest, DestroyWithoutPerformLockDoesNotReleaseMutexHeldByAnotherThread) {
    CacheGuard guard;
    auto mutex = std::make_shared<std::mutex>();
    std::atomic_int refCount{0};

    // Another thread holds the per-hash mutex for the whole duration of the test.
    std::promise<void> locked;
    std::promise<void> release;
    std::thread holder([&] {
        std::lock_guard<std::mutex> lock(*mutex);
        locked.set_value();
        release.get_future().wait();
    });
    locked.get_future().wait();

    {
        // Entry is created but destroyed before perform_lock() is called.
        CacheGuardEntry entry(guard, "hash", mutex, refCount);
        EXPECT_EQ(refCount, 1);
    }
    EXPECT_EQ(refCount, 0);

    // The mutex is still owned by 'holder' - the entry must not have released it.
    EXPECT_FALSE(mutex->try_lock());

    release.set_value();
    holder.join();

    // Once the real owner is done, the mutex becomes available as usual.
    ASSERT_TRUE(mutex->try_lock());
    mutex->unlock();
}

TEST(CacheGuardEntryTest, DestroyAfterPerformLockReleasesMutex) {
    CacheGuard guard;
    auto mutex = std::make_shared<std::mutex>();
    std::atomic_int refCount{0};

    {
        CacheGuardEntry entry(guard, "hash", mutex, refCount);
        entry.perform_lock();
        EXPECT_EQ(refCount, 1);
    }
    EXPECT_EQ(refCount, 0);

    ASSERT_TRUE(mutex->try_lock());
    mutex->unlock();
}

TEST(CacheGuardTest, HashLockIsExclusivePerHashAndReleasedOnDestruction) {
    CacheGuard guard;
    auto lock = guard.get_hash_lock("hash");

    // A different hash is not blocked by the held lock.
    EXPECT_NO_THROW(guard.get_hash_lock("other_hash"));

    std::atomic<bool> acquired{false};
    std::thread waiter([&] {
        auto other = guard.get_hash_lock("hash");
        acquired = true;
    });

    // Best effort check that the waiter is blocked while 'lock' is alive.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(acquired);

    lock.reset();
    waiter.join();
    EXPECT_TRUE(acquired);
}

}  // namespace ov::test
