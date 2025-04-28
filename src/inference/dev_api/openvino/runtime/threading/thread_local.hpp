// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief A file containing thread local class implementation.
 * @file openvino/runtime/threading/thread_local.hpp
 */

#include "openvino/core/parallel.hpp"

#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
#    include <tbb/enumerable_thread_specific.h>
#else
#    include <functional>
#    include <memory>
#    include <mutex>
#    include <thread>
#    include <unordered_map>
#    include <utility>
#endif

namespace ov {
namespace threading {

#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO

/**
 * @brief A wrapper class to keep object to be thread local.
 * @ingroup ov_dev_api_threading
 * @tparam T A type of object to keep thread local.
 */
template <typename T>
using ThreadLocal = tbb::enumerable_thread_specific<T>;

#else

template <typename T>
struct ThreadLocal {
    using Map = std::unordered_map<std::thread::id, T>;
    using Create = std::function<T()>;
    Map _map;
    mutable std::mutex _mutex;
    Create _create;

    ThreadLocal()
        : _create{[] {
              return T{};
          }} {}
    explicit ThreadLocal(const T& init)
        : _create{[init] {
              return init;
          }} {}
    ThreadLocal(ThreadLocal&& other) : _map{std::move(other._map)}, _create{std::move(other._create)} {}
    ThreadLocal& operator=(ThreadLocal&& other) {
        _map = std::move(other._map);
        _create = std::move(other._create);
        return *this;
    }
    ThreadLocal(const ThreadLocal& other) : _create(other._create) {
        std::lock_guard<std::mutex> lock{other._mutex};
        _map = other._map;
    }
    ThreadLocal& operator=(const ThreadLocal& other) {
        std::lock_guard<std::mutex> lock{other._mutex};
        _map = other._map;
        _create = other._create;
        return *this;
    }
    explicit ThreadLocal(const Create& create_) : _create{create_} {}

    T& local() {
        auto threadId = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock{_mutex};
        auto itThreadLocal = _map.find(threadId);
        if (itThreadLocal != _map.end()) {
            return itThreadLocal->second;
        } else {
            return _map.emplace(threadId, _create()).first->second;
        }
    }

    auto size() const -> decltype(_map.size()) {
        std::lock_guard<std::mutex> lock{_mutex};
        return _map.size();
    }

    // WARNING: Thread Unsafe
    template <typename It>
    struct Iterator {
        It it;
        bool operator!=(const Iterator& other) {
            return it != other.it;
        }
        Iterator& operator++() {
            ++it;
            return *this;
        }
        auto operator*() -> decltype(it->second) {
            return it->second;
        }
        auto operator-> () -> decltype(&(it->second)) {
            return &(it->second);
        }
        auto operator*() const -> decltype(it->second) {
            return it->second;
        }
        auto operator-> () const -> decltype(&(it->second)) {
            return &(it->second);
        }
    };

    auto begin() -> Iterator<decltype(_map.begin())> {
        return {_map.begin()};
    }
    auto end() -> Iterator<decltype(_map.end())> {
        return {_map.end()};
    }
    auto begin() const -> Iterator<decltype(_map.begin())> const {
        return {_map.begin()};
    }
    auto end() const -> Iterator<decltype(_map.end())> const {
        return {_map.end()};
    }

    // CombineFunc has signature T(T,T) or T(const T&, const T&)
    template <typename CombineFunc>
    T combine(CombineFunc f_combine) {
        if (begin() != end()) {
            auto ci = begin();
            T my_result = *ci;
            while (++ci != end())
                my_result = f_combine(my_result, *ci);
            return my_result;
        } else {
            return _create();
        }
    }
};

#endif

}  // namespace threading
}  // namespace ov
