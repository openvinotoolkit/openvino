// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <memory>
#include <utility>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <details/ie_exception.hpp>

namespace vpu {

//
// Handle - smart non-owning pointer, which tracks the life time of its object
//

//
// The class is not thread safe - user can't delete original object,
// while it is used via Handle in separate thread(s)
//

template <typename T>
class Handle;

class EnableHandle {
protected:
    EnableHandle() { _lifeTimeFlag = std::make_shared<int>(); }
    ~EnableHandle() = default;

    EnableHandle(const EnableHandle&) = delete;
    EnableHandle& operator=(const EnableHandle&) = delete;

    EnableHandle(EnableHandle&&) = delete;
    EnableHandle& operator=(EnableHandle&&) = delete;

private:
    std::shared_ptr<int> _lifeTimeFlag;

private:
    template <typename T>
    friend class Handle;
};

template <typename T>
class Handle final {
public:
    Handle() = default;
    ~Handle() = default;

    Handle(std::nullptr_t) {}  // NOLINT
    Handle& operator=(std::nullptr_t) {
        _ptr = nullptr;
        _lifeTimeFlag.reset();
        return *this;
    }

    Handle(const Handle&) = default;
    Handle& operator=(const Handle&) = default;

    Handle(const T* ptr) : _ptr(const_cast<T*>(ptr)) {  // NOLINT
        IE_ASSERT(_ptr != nullptr);
        _lifeTimeFlag = _ptr->_lifeTimeFlag;
        IE_ASSERT(!_lifeTimeFlag.expired());
    }

    template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    Handle(const Handle<U>& other) : _ptr(other._ptr), _lifeTimeFlag(other._lifeTimeFlag) {  // NOLINT
    }

    template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    Handle(const std::shared_ptr<U>& other) : _ptr(other.get()) {  // NOLINT
        IE_ASSERT(_ptr != nullptr);
        _lifeTimeFlag = _ptr->_lifeTimeFlag;
        IE_ASSERT(!_lifeTimeFlag.expired());
    }

    template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
    Handle(const std::unique_ptr<U>& other) : _ptr(other.get()) {  // NOLINT
        IE_ASSERT(_ptr != nullptr);
        _lifeTimeFlag = _ptr->_lifeTimeFlag;
        IE_ASSERT(!_lifeTimeFlag.expired());
    }

    bool expired() const {
        return _lifeTimeFlag.expired();
    }
    explicit operator bool() const {
        return !expired();
    }

    T* get() const {
        return expired() ? nullptr : _ptr;
    }

    T& operator*() const {
        IE_ASSERT(!expired());
        return *_ptr;
    }

    T* operator->() const {
        IE_ASSERT(!expired());
        return _ptr;
    }

    template <typename U>
    Handle<U> staticCast() const {
        return expired() ? Handle<U>() : Handle<U>(static_cast<U*>(_ptr));
    }

    template <typename U>
    Handle<U> dynamicCast() const {
        if (expired()) {
            return Handle<U>();
        }
        if (auto newPtr = dynamic_cast<U*>(_ptr)) {
            return Handle<U>(newPtr);
        }
        return Handle<U>();
    }

    bool operator<(const Handle<T>& rhs) const {
        return get() < rhs.get();
    }

private:
    T* _ptr = nullptr;
    std::weak_ptr<int> _lifeTimeFlag;

    template <typename U>
    friend class Handle;
};

template <typename T1, typename T2>
bool operator==(const Handle<T1>& first, const Handle<T2>& second) {
    return first.get() == second.get();
}
template <typename T1, typename T2>
bool operator!=(const Handle<T1>& first, const Handle<T2>& second) {
    return first.get() != second.get();
}

template <typename T1, typename T2>
bool operator==(const T1* first, const Handle<T2>& second) {
    return first == second.get();
}
template <typename T1, typename T2>
bool operator==(const Handle<T1>& first, const T2* second) {
    return first.get() == second;
}
template <typename T1, typename T2>
bool operator!=(const T1* first, const Handle<T2>& second) {
    return first != second.get();
}
template <typename T1, typename T2>
bool operator!=(const Handle<T1>& first, const T2* second) {
    return first.get() != second;
}

template <typename T1, typename T2>
bool operator==(const std::shared_ptr<T1>& first, const Handle<T2>& second) {
    return first.get() == second.get();
}
template <typename T1, typename T2>
bool operator==(const Handle<T1>& first, const std::shared_ptr<T2>& second) {
    return first.get() == second.get();
}
template <typename T1, typename T2>
bool operator!=(const std::shared_ptr<T1>& first, const Handle<T2>& second) {
    return first.get() != second.get();
}
template <typename T1, typename T2>
bool operator!=(const Handle<T1>& first, const std::shared_ptr<T2>& second) {
    return first.get() != second.get();
}

template <typename T1, typename T2>
bool operator==(const std::unique_ptr<T1>& first, const Handle<T2>& second) {
    return first.get() == second.get();
}
template <typename T1, typename T2>
bool operator==(const Handle<T1>& first, const std::unique_ptr<T2>& second) {
    return first.get() == second.get();
}
template <typename T1, typename T2>
bool operator!=(const std::unique_ptr<T1>& first, const Handle<T2>& second) {
    return first.get() != second.get();
}
template <typename T1, typename T2>
bool operator!=(const Handle<T1>& first, const std::unique_ptr<T2>& second) {
    return first.get() != second.get();
}

template <typename T>
bool operator==(std::nullptr_t, const Handle<T>& h) {
    return h.expired();
}
template <typename T>
bool operator==(const Handle<T>& h, std::nullptr_t) {
    return h.expired();
}
template <typename T>
bool operator!=(std::nullptr_t, const Handle<T>& h) {
    return !h.expired();
}
template <typename T>
bool operator!=(const Handle<T>& h, std::nullptr_t) {
    return !h.expired();
}

struct HandleHash final {
    template <typename T>
    size_t operator()(const Handle<T>& handle) const {
        assert(!handle.expired());
        return std::hash<T*>()(handle.get());
    }
};

template <typename T>
using HandleSet = std::unordered_set<Handle<T>, HandleHash>;

template <typename T, typename V>
using HandleMap = std::unordered_map<Handle<T>, V, HandleHash>;

template <typename T, typename V>
using HandleMultiMap = std::unordered_multimap<Handle<T>, V, HandleHash>;

}  // namespace vpu
