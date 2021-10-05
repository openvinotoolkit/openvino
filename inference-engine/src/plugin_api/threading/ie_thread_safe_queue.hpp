// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief A file containing thread safe queue implementation
 * @file ie_thread_safe_queue.hpp
 */

#include "ie_parallel.hpp"

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
#    include <tbb/concurrent_queue.h>
template <typename T>
using ThreadSafeQueue = tbb::concurrent_queue<T>;
template <typename T>
using ThreadSafeBoundedQueue = tbb::concurrent_bounded_queue<T>;
#else
#    include <mutex>
#    include <queue>
template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(std::move(value));
    }
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_queue.empty()) {
            value = std::move(_queue.front());
            _queue.pop();
            return true;
        } else {
            return false;
        }
    }

protected:
    std::queue<T> _queue;
    std::mutex _mutex;
};

template <typename T>
class ThreadSafeBoundedQueue {
public:
    ThreadSafeBoundedQueue() = default;
    bool try_push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_capacity) {
            _queue.push(std::move(value));
        }
        return _capacity;
    }
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_capacity && !_queue.empty()) {
            value = std::move(_queue.front());
            _queue.pop();
            return true;
        } else {
            return false;
        }
    }
    void set_capacity(std::size_t newCapacity) {
        std::lock_guard<std::mutex> lock(_mutex);
        _capacity = newCapacity;
    }

protected:
    std::queue<T> _queue;
    std::mutex _mutex;
    bool _capacity = false;
};
#endif