// Copyright (C) 2018-2021 Intel Corporation
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
#include <functional>

#include "ie_common.h"
#include "ie_so_loader.h"

namespace InferenceEngine {
namespace details {
/**
 * @brief This class is a trait class that provides a creator with a function name corresponding to the templated class
 * parameter
 */
template <class T>
class SOCreatorTrait {};

/**
 * @brief Enables only `char` or `wchar_t` template specializations
 * @tparam C A char type
 */
template <typename C>
using enableIfSupportedChar = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type;

/**
 * @brief This class instantiate object using shared library
 * @tparam T An type of object SOPointer can hold
 */
template <class T>
class SOPointer {
    template <class U>
    friend class SOPointer;

    IE_SUPPRESS_DEPRECATED_START
    struct HasRelease {
        template <typename C> static char test(decltype(&C::Release));
        template <typename C> static long test(...);
        constexpr static const bool value = sizeof(test<T>(nullptr)) == sizeof(char);
    };
    IE_SUPPRESS_DEPRECATED_END

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
    SOPointer(const std::basic_string<C> & name)
        : _so(name.c_str()) {
        Load(std::integral_constant<bool, HasRelease::value>{});
    }

    /**
     * @brief Constructs an object with existing reference
     * @brief Constructs an object with existing loader
     * @param soLoader Existing pointer to a library loader
     */
    SOPointer(const SharedObjectLoader& so, const std::shared_ptr<T>& ptr) : _so{so}, _ptr{ptr} {}

    /**
     * @brief Constructs an object with existing loader
     * @param so Existing pointer to a library loader
     */
    explicit SOPointer(const SharedObjectLoader& so)
        : _so(so) {
        Load(std::integral_constant<bool, HasRelease::value>{});
    }

    /**
     * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
     * @param that copied SOPointer object
     */
    template <typename U>
    SOPointer(const SOPointer<U>& that)
        : _so(that._so),
          _ptr(std::dynamic_pointer_cast<T>(that._ptr)) {
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

    operator const SharedObjectLoader&() const noexcept {
        return _so;
    }

    operator std::shared_ptr<T>& () noexcept {
        return _ptr;
    }

protected:
    /**
     * @brief Implements load of object from library if Release method is presented
     */
    void Load(std::true_type) {
        try {
            void* create = nullptr;
            try {
                create = _so.get_symbol((SOCreatorTrait<T>::name + std::string("Shared")).c_str());
            } catch (const NotFound&) {}
            if (create == nullptr) {
                create = _so.get_symbol(SOCreatorTrait<T>::name);
                using CreateF = StatusCode(T*&, ResponseDesc*);
                T* object = nullptr;
                ResponseDesc desc;
                StatusCode sts = reinterpret_cast<CreateF*>(create)(object, &desc);
                if (sts != OK) {
                    IE_EXCEPTION_SWITCH(sts, ExceptionType,
                        InferenceEngine::details::ThrowNow<ExceptionType>{} <<= std::stringstream{} << IE_LOCATION << desc.msg)
                }
                IE_SUPPRESS_DEPRECATED_START
                _ptr = std::shared_ptr<T>(object, [] (T* ptr){ptr->Release();});
                IE_SUPPRESS_DEPRECATED_END
            } else {
                using CreateF = void(std::shared_ptr<T>&);
                reinterpret_cast<CreateF*>(create)(_ptr);
            }
        } catch(...) {details::Rethrow();}
    }

    /**
     * @brief Implements load of object from library
     */
    void Load(std::false_type) {
        try {
            using CreateF = void(std::shared_ptr<T>&);
            reinterpret_cast<CreateF*>(_so.get_symbol(SOCreatorTrait<T>::name))(_ptr);
        } catch(...) {details::Rethrow();}
    }

    /**
     * @brief The DLL
     */
    SharedObjectLoader _so;

    /**
     * @brief Gets a smart pointer to the custom object
     */
    std::shared_ptr<T> _ptr;
};
}  // namespace details
}  // namespace InferenceEngine
