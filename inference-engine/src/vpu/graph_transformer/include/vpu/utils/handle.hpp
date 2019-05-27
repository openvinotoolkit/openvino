// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <details/ie_exception.hpp>

namespace vpu {

template <typename T>
class Handle final {
public:
    inline Handle() = default;

    inline ~Handle() = default;

    inline Handle(std::nullptr_t) {}  // NOLINT

    template <typename U>
    inline Handle(const std::shared_ptr<U>& ptr) : _weak(ptr), _plain(ptr.get()) {  // NOLINT
        IE_ASSERT(_plain != nullptr);
    }

    template <typename U>
    inline Handle(const Handle<U>& other) : _weak(other._weak), _plain(other._plain) {}  // NOLINT

    inline Handle(const Handle&) = default;
    inline Handle& operator=(const Handle&) = default;

    inline Handle(Handle&& other) : _weak(std::move(other._weak)), _plain(other._plain) {
        other._plain = nullptr;
    }
    inline Handle& operator=(Handle&& other) {
        if (&other != this) {
            _weak = std::move(other._weak);
            _plain = other._plain;
            other._plain = nullptr;
        }
        return *this;
    }

    inline Handle& operator=(std::nullptr_t) {
        _weak.reset();
        _plain = nullptr;
        return *this;
    }

    inline std::shared_ptr<T> lock() const {
        return _weak.lock();
    }

    inline bool expired() const {
        return _weak.expired();
    }

    inline T* get() const {
        return _weak.expired() ? nullptr : _plain;
    }
    inline T* getPlain() const {
        return _plain;
    }

    inline T& operator*() const {
        IE_ASSERT(!_weak.expired());
        return *_plain;
    }

    inline T* operator->() const {
        IE_ASSERT(!_weak.expired());
        return _plain;
    }

    template <typename U>
    inline Handle<U> dynamicCast() const {
        if (auto newPtr = std::dynamic_pointer_cast<U>(_weak.lock())) {
            return Handle<U>(newPtr);
        }
        return nullptr;
    }

    inline explicit operator bool() const {
        return !_weak.expired();
    }

private:
    std::weak_ptr<T> _weak;
    T* _plain = nullptr;

    template <typename U>
    friend class Handle;
};

template <typename T>
inline bool operator==(const Handle<T>& first, const Handle<T>& second) {
    return first.get() == second.get();
}
template <typename T>
inline bool operator!=(const Handle<T>& first, const Handle<T>& second) {
    return first.get() != second.get();
}
template <typename T>
inline bool operator==(const Handle<T>& first, const std::shared_ptr<T>& second) {
    return first.get() == second.get();
}
template <typename T>
inline bool operator!=(const Handle<T>& first, const std::shared_ptr<T>& second) {
    return first.get() != second.get();
}
template <typename T>
inline bool operator==(std::nullptr_t, const Handle<T>& h) {
    return h.get() == nullptr;
}
template <typename T>
inline bool operator==(const Handle<T>& h, std::nullptr_t) {
    return h.get() == nullptr;
}
template <typename T>
inline bool operator!=(std::nullptr_t, const Handle<T>& h) {
    return h.get() != nullptr;
}
template <typename T>
inline bool operator!=(const Handle<T>& h, std::nullptr_t) {
    return h.get() != nullptr;
}

struct HandleHash final {
    template <typename T>
    inline size_t operator()(const Handle<T>& handle) const {
        return std::hash<T*>()(handle.getPlain());
    }
};

template <class Base>
class EnableHandleFromThis : public std::enable_shared_from_this<Base> {
public:
    inline Handle<Base> handle_from_this() const {
        return Handle<Base>(std::const_pointer_cast<Base>(this->shared_from_this()));
    }

protected:
    inline EnableHandleFromThis() = default;
    inline EnableHandleFromThis(const EnableHandleFromThis&) = default;
    inline EnableHandleFromThis(EnableHandleFromThis&&) = default;
    inline ~EnableHandleFromThis() = default;
    inline EnableHandleFromThis& operator=(const EnableHandleFromThis&) = default;
    inline EnableHandleFromThis& operator=(EnableHandleFromThis&&) = default;
};

}  // namespace vpu
