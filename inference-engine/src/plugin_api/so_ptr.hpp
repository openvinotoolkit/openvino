// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a wrapper class for handling plugin instantiation and releasing resources
 * @file so_ptr.hpp
 */
#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#include "openvino/runtime/common.hpp"

namespace ov {
namespace runtime {

/**
 * @brief This class instantiate object using shared library
 * @tparam T An type of object SoPtr can hold
 */
template <class T>
struct SoPtr {
    /**
     * @brief Default constructor
     */
    SoPtr() = default;

    /**
     * @brief Constructs an object with existing shared object reference and loaded pointer
     * @param soLoader Existing pointer to a library loader
     */
    SoPtr(const std::shared_ptr<void>& so, const std::shared_ptr<T>& ptr) : _so{so}, _ptr{ptr} {}

    /**
     * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
     * @param that copied SoPtr object
     */
    template <typename U>
    SoPtr(const SoPtr<U>& that) : _so{that._so},
                                  _ptr{std::dynamic_pointer_cast<T>(that._ptr)} {
        IE_ASSERT(_ptr != nullptr);
    }

    /**
     * @brief Standard pointer operator
     * @return underlined interface with disabled Release method
     */
    T* operator->() const noexcept {
        return _ptr.get();
    }

    explicit operator bool() const noexcept {
        return _ptr != nullptr;
    }

    friend bool operator==(std::nullptr_t, const SoPtr& ptr) noexcept {
        return !ptr;
    }
    friend bool operator==(const SoPtr& ptr, std::nullptr_t) noexcept {
        return !ptr;
    }
    friend bool operator!=(std::nullptr_t, const SoPtr& ptr) noexcept {
        return static_cast<bool>(ptr);
    }
    friend bool operator!=(const SoPtr& ptr, std::nullptr_t) noexcept {
        return static_cast<bool>(ptr);
    }

    /**
     * @brief The shared object or dinamic loaded library
     */
    std::shared_ptr<void> _so;

    /**
     * @brief Gets a smart pointer to the custom object
     */
    std::shared_ptr<T> _ptr;
};
}  // namespace runtime
}  // namespace ov
