// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief This is a wrapper class for handling plugin instantiation and releasing resources
* @file ie_so_pointer.hpp
*/
#pragma once

#include <memory>
#include "ie_so_loader.h"
#include "ie_common.h"
#include "ie_plugin.hpp"
#include "details/ie_exception.hpp"
#include "details/ie_no_release.hpp"
#include <string>
#include <cassert>

namespace InferenceEngine {
namespace details {

/**
* @brief This class is a C++ helper to load a symbol from a library and create its instance
*/
template<class Loader>
class SymbolLoader {
private:
    std::shared_ptr<Loader> _so_loader;

public:
    /**
    * @brief The main constructor
    * @param loader Library to load from
    */
    explicit SymbolLoader(std::shared_ptr<Loader> loader) : _so_loader(loader) {}

    /**
    * @brief Calls a function from the library that creates an object and returns StatusCode
    * @param name Name of function to load object with
    * @return If StatusCode provided by function is OK then returns the loaded object. Throws an exception otherwise
    */
    template<class T>
    T* instantiateSymbol(const std::string& name) const {
        T* instance = nullptr;
        ResponseDesc desc;
        StatusCode sts = bind_function<StatusCode(T*&, ResponseDesc*)>(name)(instance, &desc);
        if (sts != OK) {
            THROW_IE_EXCEPTION << desc.msg;
        }
        return instance;
    }

    /**
    * @brief Loads function from the library and returns a pointer to it
    * @param functionName Name of function to load
    * @return The loaded function
    */
    template<class T>
    std::function<T> bind_function(const std::string &functionName) const {
        std::function<T> ptr(reinterpret_cast<T *>(_so_loader->get_symbol(functionName.c_str())));
        return ptr;
    }
};

/**
* @brief This class is a trait class that provides a creator with a function name corresponding to the templated class parameter
*/
template<class T>
class SOCreatorTrait {};

/**
* @brief This class instantiate object using shared library
*/
template <class T, class Loader = SharedObjectLoader>
class SOPointer {
IE_SUPPRESS_DEPRECATED_START
    template <class U, class W> friend class SOPointer;
public:
    /**
    * @brief Default constructor
    */
    SOPointer() = default;

    /**
    * @brief The main constructor
    * @param name Name of a shared library file
    */
    explicit SOPointer(const file_name_t &name)
        : _so_loader(new Loader(name.c_str()))
        , _pointedObj(details::shared_from_irelease(
            SymbolLoader<Loader>(_so_loader).template instantiateSymbol<T>(SOCreatorTrait<T>::name))) {
    }

    /**
    * @brief Constructs an object with existing reference
    * @param _pointedObj_ Existing reference to wrap
    */
    explicit SOPointer(T * _pointedObj_)
        : _so_loader()
        , _pointedObj(_pointedObj_) {
        if (_pointedObj == nullptr) {
            THROW_IE_EXCEPTION << "Cannot create SOPointer<T, Loader> from nullptr";
        }
    }

    /**
    * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
    * @param that copied SOPointer object
    */
    template<class U, class W>
    SOPointer(const SOPointer<U, W> & that) :
        _so_loader(std::dynamic_pointer_cast<Loader>(that._so_loader)),
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

    friend bool operator == (std::nullptr_t, const SOPointer& ptr) noexcept {
        return !ptr;
    }
    friend bool operator == (const SOPointer& ptr, std::nullptr_t) noexcept {
        return !ptr;
    }
    friend bool operator != (std::nullptr_t, const SOPointer& ptr) noexcept {
        return static_cast<bool>(ptr);
    }
    friend bool operator != (const SOPointer& ptr, std::nullptr_t) noexcept {
        return static_cast<bool>(ptr);
    }

    SOPointer& operator=(const SOPointer & pointer) noexcept {
        _pointedObj =  pointer._pointedObj;
        _so_loader = pointer._so_loader;
        return *this;
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
 * @param name Name of the shared library file
 */
template <class T>
inline std::shared_ptr<T> make_so_pointer(const file_name_t & name) = delete;

}  // namespace InferenceEngine
