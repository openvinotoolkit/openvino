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

#include "details/ie_exception.hpp"
#include "details/ie_no_release.hpp"
#include "details/ie_irelease.hpp"
#include "details/os/os_filesystem.hpp"
#include "ie_common.h"
#include "ie_so_loader.h"

namespace InferenceEngine {
namespace details {

/**
 * @brief This class is a C++ helper to load a symbol from a library and create its instance
 */
template <class Loader>
class SymbolLoader {
private:
    std::shared_ptr<Loader> _so_loader;

public:
    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief The main constructor
     * @param loader Library to load from
     */
    explicit SymbolLoader(std::shared_ptr<Loader> loader): _so_loader(loader) {
        if (_so_loader == nullptr) {
            THROW_IE_EXCEPTION << "SymbolLoader cannot be created with nullptr";
        }
    }

    /**
     * @brief Calls a function from the library that creates an object and returns StatusCode
     * @param name Name of function to load object with
     * @return If StatusCode provided by function is OK then returns the loaded object. Throws an exception otherwise
     */
    template <class T>
    T* instantiateSymbol(const std::string& name) const {
        IE_SUPPRESS_DEPRECATED_START
        T* instance = nullptr;
        ResponseDesc desc;
        StatusCode sts = bind_function<StatusCode(T*&, ResponseDesc*)>(name)(instance, &desc);
        if (sts != OK) {
            THROW_IE_EXCEPTION << desc.msg;
        }
        return instance;
        IE_SUPPRESS_DEPRECATED_END
    }

private:
    /**
     * @brief Loads function from the library and returns a pointer to it
     * @param functionName Name of function to load
     * @return The loaded function
     */
    template <class T>
    std::function<T> bind_function(const std::string& functionName) const {
        std::function<T> ptr(reinterpret_cast<T*>(_so_loader->get_symbol(functionName.c_str())));
        return ptr;
    }

    IE_SUPPRESS_DEPRECATED_END
};

/**
 * @brief This class is a trait class that provides a creator with a function name corresponding to the templated class
 * parameter
 */
template <class T>
class SOCreatorTrait {};

/**
 * @brief This class instantiate object using shared library
 * @tparam T An type of object SOPointer can hold
 * @tparam Loader A loader used to load a library
 */
template <class T, class Loader = SharedObjectLoader>
class SOPointer {
    IE_SUPPRESS_DEPRECATED_START
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
        : _so_loader(new Loader(name.c_str())),
          _pointedObj(details::shared_from_irelease(
              SymbolLoader<Loader>(_so_loader).template instantiateSymbol<T>(SOCreatorTrait<T>::name))) {}

    /**
     * @brief The main constructor
     * @param name Name of a shared library file
     */
    explicit SOPointer(const char * name)
        : _so_loader(new Loader(name)),
          _pointedObj(details::shared_from_irelease(
              SymbolLoader<Loader>(_so_loader).template instantiateSymbol<T>(SOCreatorTrait<T>::name))) {}

    /**
     * @brief Constructs an object with existing reference
     * @param pointedObj Existing reference to wrap
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
    explicit SOPointer(std::shared_ptr<Loader> so_loader)
        : _so_loader(so_loader),
          _pointedObj(details::shared_from_irelease(
              SymbolLoader<Loader>(_so_loader).template instantiateSymbol<T>(SOCreatorTrait<T>::name))) {}

    /**
     * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
     * @param that copied SOPointer object
     */
    template <class U, class W>
    SOPointer(const SOPointer<U, W>& that)
        : _so_loader(std::dynamic_pointer_cast<Loader>(that._so_loader)),
          _pointedObj(std::dynamic_pointer_cast<T>(that._pointedObj)) {
        if (_pointedObj == nullptr) {
            THROW_IE_EXCEPTION << "Cannot create object from SOPointer<U, W> reference";
        }
    }

    /**
     * @brief Standard pointer operator
     * @return underlined interface with disabled Release method
     */
    details::NoReleaseOn<T>* operator->() const noexcept {
        return reinterpret_cast<details::NoReleaseOn<T>*>(_pointedObj.get());
    }

    /**
     * @brief Standard dereference operator
     * @return underlined interface with disabled Release method
     */
    details::NoReleaseOn<T>* operator*() const noexcept {
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
    IE_SUPPRESS_DEPRECATED_END
};

}  // namespace details

/**
 * @brief Creates a special shared_pointer wrapper for the given type from a specific shared module
 * @tparam T An type of object SOPointer can hold
 * @param name Name of the shared library file
 * @return A created object
 */
template <class T>
inline std::shared_ptr<T> make_so_pointer(const file_name_t& name) = delete;

}  // namespace InferenceEngine
