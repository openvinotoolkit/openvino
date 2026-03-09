// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a wrapper class for handling plugin instantiation and releasing resources
 * @file openvino/runtime/so_ptr.hpp
 */
#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#include "openvino/runtime/common.hpp"

namespace ov {

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

    SoPtr(const SoPtr<T>&) = default;
    SoPtr(SoPtr<T>&&) noexcept = default;

    SoPtr& operator=(const SoPtr<T>&) = default;
    SoPtr& operator=(SoPtr<T>&&) noexcept = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~SoPtr() {
        _ptr = {};
    }

    /**
     * @brief Constructs an object with existing shared object reference and loaded pointer
     * @param ptr pointer to the loaded object
     * @param so Existing reference to library
     */
    SoPtr(const std::shared_ptr<T>& ptr, const std::shared_ptr<void>& so) : _ptr{ptr}, _so{so} {}

    /**
     * @brief Constructs an object with existing shared object reference
     * @param ptr pointer to the loaded object
     */
    SoPtr(const std::shared_ptr<T>& ptr) : _ptr{ptr}, _so{nullptr} {}

    /**
     * @brief Constructs an object with existing shared object reference
     * @param ptr pointer to the loaded object
     */
    template <class U, std::enable_if_t<std::is_base_of_v<T, U>, bool> = true>
    SoPtr(const std::shared_ptr<U>& ptr) : _ptr{std::static_pointer_cast<T>(ptr)},
                                           _so{nullptr} {}

    /**
     * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
     * @param that copied SoPtr object
     */
    template <typename U>
    SoPtr(const SoPtr<U>& that) : _ptr{std::dynamic_pointer_cast<T>(that._ptr)},
                                  _so{that._so} {
        OPENVINO_ASSERT(_ptr != nullptr);
    }

    /**
     * @brief Standard pointer operator
     * @return underlined interface with disabled Release method
     */
    T* operator->() const noexcept {
        return _ptr.get();
    }

    /**
     * @brief Dereference stored pointer to T object.
     * @return Reference to T object.
     */
    T& operator*() const noexcept {
        return *_ptr;
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
     * @brief Gets a smart pointer to the custom object
     */
    std::shared_ptr<T> _ptr;

    /**
     * @brief The shared object or dynamic loaded library
     */
    std::shared_ptr<void> _so;
};

}  // namespace ov
