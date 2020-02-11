// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <type_traits>

#include <details/ie_exception.hpp>

namespace vpu {

template <typename T>
class Optional final {
public:
    Optional() noexcept : _mem{} {
    }

    ~Optional() {
        reset();
    }

    Optional(const Optional& other) : _mem{} {
        if (other._hasValue) {
            constructValue(other.getValueRef());
        }
    }
    Optional& operator=(const Optional& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueRef();
                } else {
                    constructValue(other.getValueRef());
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    Optional(Optional&& other) : _mem{} {
        if (other._hasValue) {
            constructValue(other.getValueMoveRef());
        }
    }
    Optional& operator=(Optional&& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueMoveRef();
                } else {
                    constructValue(other.getValueMoveRef());
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    template <typename U>
    Optional(const Optional<U>& other) : _mem{} {
        if (other._hasValue) {
            constructValue(other.getValueRef());
        }
    }
    template <typename U>
    Optional& operator=(const Optional<U>& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueRef();
                } else {
                    constructValue(other.getValueRef());
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    template <typename U>
    Optional(Optional<U>&& other) : _mem{} {
        if (other._hasValue) {
            constructValue(other.getValueMoveRef());
        }
    }
    template <typename U>
    Optional& operator=(Optional<U>&& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueMoveRef();
                } else {
                    constructValue(other.getValueMoveRef());
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    Optional(const T& value) : _mem{} {  // NOLINT
        constructValue(value);
    }
    Optional& operator=(const T& value) {
        if (_hasValue) {
            getValueRef() = value;
        } else {
            constructValue(value);
        }
        return *this;
    }


    Optional(T&& value) : _mem{} {  // NOLINT
        constructValue(std::move(value));
    }
    Optional& operator=(T&& value) {
        if (_hasValue) {
            getValueRef() = std::move(value);
        } else {
            constructValue(std::move(value));
        }
        return *this;
    }

    template <typename... Args, typename = typename std::enable_if<std::is_constructible<T, Args...>::value>::type>
    Optional(Args&&... args) : _mem{} {  // NOLINT
        constructValue(std::forward<Args>(args)...);
    }

    void reset() noexcept {
        if (_hasValue) {
            destroyValue();
        }
    }

    bool hasValue() const noexcept {
        return _hasValue;
    }

    const T& get() const {
        IE_ASSERT(_hasValue);
        return getValueRef();
    }

    T getOrDefault() const {
        if (_hasValue) {
            return getValueRef();
        } else {
            return T();
        }
    }
    T getOrDefault(const T& def) const {
        if (_hasValue) {
            return getValueRef();
        } else {
            return def;
        }
    }
    T getOrDefault(T&& def) const {
        if (_hasValue) {
            return getValueRef();
        } else {
            return def;
        }
    }

    T&& move() {
        IE_ASSERT(_hasValue);
        return getValueMoveRef();
    }

private:
    using Memory = typename std::aligned_storage<sizeof(T), alignof(T)>::type[1];

    T& getValueRef() {
        return *reinterpret_cast<T*>(_mem);
    }
    const T& getValueRef() const {
        return *reinterpret_cast<const T*>(_mem);
    }

    T&& getValueMoveRef() {
        return std::move(getValueRef());
    }

    template <typename... Args, typename = typename std::enable_if<std::is_constructible<T, Args...>::value>::type>
    void constructValue(Args&&... args) {
        new (_mem) T(std::forward<Args>(args)...);
        _hasValue = true;
    }

    void destroyValue() {
        reinterpret_cast<T*>(_mem)->~T();
        _hasValue = false;
    }

private:
    // TODO: actually, it would be better to initialize _mem here instead of doing it
    // in each contructor but it causes a segfault in gcc 4.8
    Memory _mem;
    bool _hasValue = false;

    template <typename U>
    friend class Optional;
};

}  // namespace vpu
