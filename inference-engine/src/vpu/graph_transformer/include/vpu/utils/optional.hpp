// Copyright (C) 2018-2019 Intel Corporation
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
    inline Optional() noexcept : _mem{}, _hasValue(false) {
    }

    inline ~Optional() {
        reset();
    }

    inline Optional(const Optional& other) : _mem{}, _hasValue(false) {
        if (other._hasValue) {
            constructValue(other.getValueRef());
            _hasValue = true;
        }
    }
    inline Optional& operator=(const Optional& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueRef();
                } else {
                    constructValue(other.getValueRef());
                    _hasValue = true;
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    inline Optional(Optional&& other) : _mem{}, _hasValue(false) {
        if (other._hasValue) {
            constructValue(other.getValueMoveRef());
            _hasValue = true;
        }
    }
    inline Optional& operator=(Optional&& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueMoveRef();
                } else {
                    constructValue(other.getValueMoveRef());
                    _hasValue = true;
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    template <typename U>
    inline Optional(const Optional<U>& other) : _mem{}, _hasValue(false) {
        if (other._hasValue) {
            constructValue(other.getValueRef());
            _hasValue = true;
        }
    }
    template <typename U>
    inline Optional& operator=(const Optional<U>& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueRef();
                } else {
                    constructValue(other.getValueRef());
                    _hasValue = true;
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    template <typename U>
    inline Optional(Optional<U>&& other) : _mem{}, _hasValue(false) {
        if (other._hasValue) {
            constructValue(other.getValueMoveRef());
            _hasValue = true;
        }
    }
    template <typename U>
    inline Optional& operator=(Optional<U>&& other) {
        if (this != &other) {
            if (other._hasValue) {
                if (_hasValue) {
                    getValueRef() = other.getValueMoveRef();
                } else {
                    constructValue(other.getValueMoveRef());
                    _hasValue = true;
                }
            } else {
                reset();
            }
        }
        return *this;
    }

    template <typename U>
    inline Optional(U&& value) : _mem{}, _hasValue(true) {  // NOLINT
        constructValue(std::forward<U>(value));
    }

    template <typename U>
    inline Optional& operator=(U&& value) {
        if (_hasValue) {
            getValueRef() = std::forward<U>(value);
        } else {
            constructValue(std::forward<U>(value));
            _hasValue = true;
        }
        _hasValue = true;
        return *this;
    }

    inline void reset() noexcept {
        if (_hasValue) {
            destroyValue();
            _hasValue = false;
        }
    }

    inline bool hasValue() const noexcept {
        return _hasValue;
    }

    inline const T& get() const {
        IE_ASSERT(_hasValue);
        return getValueRef();
    }

    template <typename U>
    inline T getOrDefault(U&& def) const {
        if (_hasValue) {
            return getValueRef();
        } else {
            return std::forward<U>(def);
        }
    }

private:
    using Memory = typename std::aligned_storage<sizeof(T), alignof(T)>::type[1];

    inline T& getValueRef() {
        return *reinterpret_cast<T*>(_mem);
    }
    inline const T& getValueRef() const {
        return *reinterpret_cast<const T*>(_mem);
    }

    inline T&& getValueMoveRef() {
        return std::move(getValueRef());
    }

    template <typename U>
    inline void constructValue(U&& value) {
        new (_mem) T(std::forward<U>(value));
    }

    inline void destroyValue() {
        reinterpret_cast<T*>(_mem)->~T();
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
