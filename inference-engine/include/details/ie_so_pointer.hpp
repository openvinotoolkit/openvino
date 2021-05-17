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
class SOCreatorTrait {
public:
    /**
     * @brief A dummy name for fabric
     */
    static constexpr auto name = "";
};

/**
 * @brief Enables only `char` or `wchar_t` template specializations
 * @tparam C A char type
 */
template <typename C>
using enableIfSupportedChar = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type;

/// @cond
IE_SUPPRESS_DEPRECATED_START
template <class T>
struct HasRelease {
    template <typename C> static char test(decltype(&C::Release));
    template <typename C> static long test(...);
    constexpr static const bool value = sizeof(test<T>(nullptr)) == sizeof(char);
};
IE_SUPPRESS_DEPRECATED_END
/// @endcond

/**
 * @brief This class instantiate object loaded from the shared library
 * @tparam T An type of object SOPointer can hold
 */
template <class T, bool = HasRelease<T>::value>
class SOPointer;

#define CATCH_IE_EXCEPTION(ExceptionType) catch (const InferenceEngine::ExceptionType& e) {throw e;}
#define CATCH_IE_EXCEPTIONS                     \
        CATCH_IE_EXCEPTION(GeneralError)        \
        CATCH_IE_EXCEPTION(NotImplemented)      \
        CATCH_IE_EXCEPTION(NetworkNotLoaded)    \
        CATCH_IE_EXCEPTION(ParameterMismatch)   \
        CATCH_IE_EXCEPTION(NotFound)            \
        CATCH_IE_EXCEPTION(OutOfBounds)         \
        CATCH_IE_EXCEPTION(Unexpected)          \
        CATCH_IE_EXCEPTION(RequestBusy)         \
        CATCH_IE_EXCEPTION(ResultNotReady)      \
        CATCH_IE_EXCEPTION(NotAllocated)        \
        CATCH_IE_EXCEPTION(InferNotStarted)     \
        CATCH_IE_EXCEPTION(NetworkNotRead)      \
        CATCH_IE_EXCEPTION(InferCancelled)

/**
 * @brief This class instantiate object loaded from the shared library and use Release method to destroy the object
 * @tparam T An type of object SOPointer can hold
 */
template <class T>
class SOPointer<T, true> {
    template <class U, bool>
    friend class SOPointer;

public:
    /**
     * @brief Default constructor
     */
    SOPointer() = default;

    /**
     * @brief Constructs an object using name to load library
     * @param name Existing pointer to a library loader
     */
    template <typename C>
    SOPointer(const std::basic_string<C>& name) :
        SOPointer(details::SharedObjectLoader(name.c_str())) {
    }

    /**
     * @brief Constructs an object with existing loader
     * @param so_loader Existing pointer to a library loader
     */
    SOPointer(const SharedObjectLoader& so_loader)
        : _so{so_loader} {
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
                _ptr = std::shared_ptr<T>{object, [] (T* ptr){ptr->Release();}};
                IE_SUPPRESS_DEPRECATED_END
            } else {
                using CreateF = void(std::shared_ptr<T>&);
                reinterpret_cast<CreateF*>(create)(_ptr);
            }
        } CATCH_IE_EXCEPTIONS catch (const std::exception& ex) {
            IE_THROW() << ex.what();
        } catch(...) {
            IE_THROW(Unexpected);
        }
    }

    /**
     * @brief Constructs an object with existing loader
     * @param so Existing pointer to a library loader
     * @param ptr A shared pointer to the object
     */
    SOPointer(const SharedObjectLoader& so, const std::shared_ptr<T>& ptr) : _so{so}, _ptr{ptr} {}

    /**
     * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
     * @param that copied SOPointer object
     */
    template <class U>
    SOPointer(const SOPointer<U, true>& that)
        : _so(that._so),
          _ptr(std::dynamic_pointer_cast<T>(that._ptr)) {}

    /**
     * @brief Standard pointer operator
     */
    T* operator->() const noexcept {
        return _ptr.get();
    }

    /**
     * @return raw pointer
     */
    T* get() noexcept {
        return _ptr.get();
    }

    /**
     * @return raw pointer
     */
    const T* get() const noexcept {
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
    SharedObjectLoader _so;
    std::shared_ptr<T> _ptr;
};

/**
 * @brief This class instantiate object loaded from the shared library
 * @tparam T An type of object SOPointer can hold
 */
template <class T>
class SOPointer<T, false> {
    template <class U, bool>
    friend class SOPointer;

public:
    /**
     * @brief Default constructor
     */
    SOPointer() = default;

    /**
     * @brief Constructs an object using name to load library
     * @param name Existing pointer to a library loader
     */
    template <typename C, typename = enableIfSupportedChar<C>>
    SOPointer(const std::basic_string<C>& name) :
        SOPointer(details::SharedObjectLoader(name.c_str())) {
    }

    SOPointer(const SharedObjectLoader& so_loader)
    : _so{so_loader} {
        try {
            using CreateF = void(std::shared_ptr<T>&);
            reinterpret_cast<CreateF*>(_so.get_symbol(SOCreatorTrait<T>::name))(_ptr);
        } CATCH_IE_EXCEPTIONS catch (const std::exception& ex) {
            IE_THROW() << ex.what();
        } catch(...) {
            IE_THROW(Unexpected);
        }
    }

    /**
     * @brief Constructs an object with existing loader
     * @param soLoader Existing pointer to a library loader
     */
    SOPointer(const SharedObjectLoader& so, const std::shared_ptr<T>& ptr) : _so{so}, _ptr{ptr} {}

    /**
     * @brief The copy-like constructor, can create So Pointer that dereferenced into child type if T is derived of U
     * @param that copied SOPointer object
     */
    template <class U>
    SOPointer(const SOPointer<U, false>& that)
        : _so(that._so),
          _ptr(std::dynamic_pointer_cast<T>(that._ptr)) {}

    /**
     * @brief Standard pointer operator
     */
    T* operator->() const noexcept {
        return _ptr.get();
    }

    /**
     * @brief Standard dereference operator
     */
    const T* operator*() const noexcept {
        return this->operator->();
    }

    /**
     * @return raw pointer
     */
    T* get() noexcept {
        return _ptr.get();
    }

    /**
     * @return raw pointer
     */
    const T* get() const noexcept {
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
    SharedObjectLoader _so;
    std::shared_ptr<T> _ptr;
};

#undef CATCH_IE_EXCEPTION
#undef CATCH_IE_EXCEPTIONS

}  // namespace details
}  // namespace InferenceEngine
