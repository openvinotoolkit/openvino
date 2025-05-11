// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <queue>
#include <type_traits>

#include "openvino/core/parallel.hpp"

#if ((OV_THREAD == OV_THREAD_TBB) || (OV_THREAD == OV_THREAD_TBB_AUTO))
#    include <tbb/concurrent_priority_queue.h>
#    include <tbb/concurrent_queue.h>
#endif

namespace ov {
namespace threading {

template <typename T>
class ThreadSafeQueueWithSize {
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
    size_t size() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

protected:
    std::queue<T> _queue;
    std::mutex _mutex;
};
#if ((OV_THREAD == OV_THREAD_TBB) || (OV_THREAD == OV_THREAD_TBB_AUTO))
template <typename T>
using ThreadSafeQueue = tbb::concurrent_queue<T>;
template <typename T>
using ThreadSafeBoundedQueue = tbb::concurrent_bounded_queue<T>;
template <typename T>
class ThreadSafeBoundedPriorityQueue {
public:
    ThreadSafeBoundedPriorityQueue() = default;
    bool try_push(T&& value) {
        if (_capacity) {
            _pqueue.push(std::move(value));
            return true;
        }
        return false;
    }
    bool try_pop(T& value) {
        return _capacity ? _pqueue.try_pop(value) : false;
    }
    void set_capacity(std::size_t newCapacity) {
        _capacity = newCapacity;
    }

protected:
    tbb::concurrent_priority_queue<T, std::greater<T>> _pqueue;
    std::atomic_bool _capacity{false};
};
#else
template <typename T>
using ThreadSafeQueue = ThreadSafeQueueWithSize<T>;
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
template <typename T>
class ThreadSafeBoundedPriorityQueue {
public:
    ThreadSafeBoundedPriorityQueue() = default;
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
            value = std::move(_queue.top());
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
    std::priority_queue<T, std::vector<T>, std::greater<T>> _queue;
    std::mutex _mutex;
    bool _capacity = false;
};
#endif
}  // namespace threading
}  // namespace ov