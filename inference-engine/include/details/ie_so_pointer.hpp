// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a wrapper class for handling plugin instantiation and releasing resources
 * @file ie_so_pointer.hpp
 */
#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>

#include "ie_common.h"
#include "ie_so_loader.h"
#include "details/ie_exception.hpp"

namespace InferenceEngine {
namespace details {
/**
 * @brief This class is a trait class that provides a creator with a function name corresponding to the templated class
 * parameter
 */
template <class T>
class SOCreatorTrait {};

template <class T>
using CreateF = void(std::shared_ptr<T>*);

/**
 * @brief Enables only `char` or `wchar_t` template specializations
 * @tparam C A char type
 */
template <typename C>
using enableIfSupportedChar = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type;

/**
 * @brief This class instantiate object using shared library
 * @tparam T An type of object SOPointer can hold
 * @tparam Loader A loader used to load a library
 */
template <class T, class Loader = SharedObjectLoader>
class SOPointer {
    template <class U, class W>
    friend class SOPointer;

public:
    /**
     * @brief Default constructor
     */
    SOPointer() = default;

    /**
     * @brief The main constructor
     * @param name Name of a shared library file
     */
    template <typename C,
              typename = enableIfSupportedChar<C>>
    explicit SOPointer(const std::basic_string<C> & name)
        : _so_loader(new Loader(name.c_str())) {
            reinterpret_cast<CreateF<T>*>(_so_loader->get_symbol(SOCreatorTrait<T>::name))(&_pointedObj);
        }

    /**
     * @brief The main constructor
     * @param name Name of a shared library file
     */
    explicit SOPointer(const char * name)
        : _so_loader(new Loader(name)) {
            reinterpret_cast<CreateF<T>*>(_so_loader->get_symbol(SOCreatorTrait<T>::name))(&_pointedObj);
        }

    /**
     * @brief Constructs an object with existing reference
     * @param pointedObj Existing reference to  wrap
     */
    explicit SOPointer(T* pointedObj): _so_loader(), _pointedObj(pointedObj) {
        if (_pointedObj == nullptr) {
            THROW_IE_EXCEPTION << "Cannot create SOPointer<T, Loader> from nullptr";
        }
    }

    /**
     * @brief Constructs an object with existing loader
     * @param so_loader Existing pointer to a library loader
     */
    explicit SOPointer(const std::shared_ptr<Loader>& so_loader)
        : _so_loader(so_loader) {
            reinterpret_cast<CreateF<T>*>(_so_loader->get_symbol(SOCreatorTrait<T>::name))(&_pointedObj);
        }

    /**
     * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
     * @param that copied SOPointer object
     */
    template <class U, class W>
    SOPointer(const SOPointer<U, W>& that)
        : _so_loader(std::dynamic_pointer_cast<Loader>(that._so_loader)),
          _pointedObj(std::dynamic_pointer_cast<T>(that._pointedObj)) {
        IE_ASSERT(_pointedObj != nullptr);
    }

    /**
     * @brief Standard pointer operator
     * @return underlined interface with disabled Release method
     */
    T* operator->() const noexcept {
        return _pointedObj.get();
    }

    /**
     * @brief Standard dereference operator
     * @return underlined interface with disabled Release method
     */
    const T* operator*() const noexcept {
        return this->operator->();
    }

    explicit operator bool() const noexcept {
        return (nullptr != _so_loader) && (nullptr != _pointedObj);
    }

    friend bool operator==(std::nullptr_t, const SOPointer& ptr) noexcept {
        return !ptr;
    }
    friend bool operator==(const SOPointer& ptr, std::nullptr_t) noexcept {
        return !ptr;
    }
    friend bool operator!=(std::nullptr_t, const SOPointer& ptr) noexcept {
        return static_cast<bool>(ptr);
    }
    friend bool operator!=(const SOPointer& ptr, std::nullptr_t) noexcept {
        return static_cast<bool>(ptr);
    }

    SOPointer& operator=(const SOPointer& pointer) noexcept {
        _pointedObj = pointer._pointedObj;
        _so_loader = pointer._so_loader;
        return *this;
    }

    operator std::shared_ptr<Loader>() const noexcept {
        return _so_loader;
    }

protected:
    /**
     * @brief Gets a smart pointer to the DLL
     */
    std::shared_ptr<Loader> _so_loader;

    /**
     * @brief Gets a smart pointer to the custom object
     */
    std::shared_ptr<T> _pointedObj;
};
}  // namespace details
}  // namespace InferenceEngine
