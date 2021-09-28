// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ostream>
#include <typeinfo>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"

namespace ov {
struct OPENVINO_API SharedAnyImpl : std::enable_shared_from_this<SharedAnyImpl> {
    using Ptr = std::shared_ptr<SharedAnyImpl>;
    virtual const std::type_info& get_type_info() const = 0;
    virtual void print(std::ostream& os) const = 0;
    virtual void* get() = 0;
    const void* get() const {
        return const_cast<SharedAnyImpl*>(this)->get();
    }
    virtual bool equal(const SharedAnyImpl& other) const = 0;

protected:
    ~SharedAnyImpl() = default;
};

class SharedAny {
    template <typename...>
    using void_t = void;

    template <typename, typename = void>
    struct ostreamable : std::false_type {};

    // specialized as has_member< T , void > or discarded (sfinae)
    template <typename T>
    struct ostreamable<T, void_t<decltype(std::declval<std::ostream>() << std::declval<T>())>> : std::true_type {};

    template <typename, typename = void>
    struct equality_comparable : std::false_type {};

    // specialized as has_member< T , void > or discarded (sfinae)
    template <typename T>
    struct equality_comparable<T, void_t<decltype(std::declval<T>() == std::declval<T>())>> : std::true_type {};

    template <typename T>
    struct Impl : public SharedAnyImpl {
        template <typename... Args>
        Impl(Args&&... args) : value{std::forward<Args>(args)...} {}

        const std::type_info& get_type_info() const override {
            return typeid(value);
        }

        template <class U>
        static typename std::enable_if<ostreamable<U>::value>::type print(std::ostream& os, const U& value) {
            os << value;
        }

        template <class U>
        [[noreturn]] static typename std::enable_if<!ostreamable<U>::value>::type print(std::ostream&, const U&) {
            OPENVINO_UNREACHABLE(typeid(T).name(), " has no output to std::ostream operator");
        }
        void print(std::ostream& os) const override {
            print(os, value);
        }
        void* get() override {
            return &value;
        }

        template <class U>
        static typename std::enable_if<equality_comparable<U>::value, bool>::type equal(const U& rhs, const U& lhs) {
            return rhs == lhs;
        }
        template <class U>
        [[noreturn]] static typename std::enable_if<!equality_comparable<U>::value, bool>::type equal(const U&,
                                                                                                      const U&) {
            OPENVINO_UNREACHABLE(typeid(T).name(), " is not equality comparable");
        }
        bool equal(const SharedAnyImpl& other) const override {
            if (get_type_info() == other.get_type_info()) {
                return equal(this->value, *static_cast<const T*>(other.get()));
            }
            return false;
        }
        T value;
    };

public:
    SharedAny() = default;

    SharedAny(const SharedAnyImpl::Ptr& impl_) : impl{impl_} {}

    template <
        typename T,
        typename DecayT = typename std::decay<T>::type,
        typename Impl = Impl<DecayT>,
        typename std::enable_if<std::is_copy_constructible<DecayT>::value && !std::is_same<DecayT, SharedAny>::value,
                                bool>::type = true>
    SharedAny(T&& value) : impl{std::make_shared<Impl>(std::forward<T>(value))} {}

    template <typename T, typename... Args>
    static SharedAny make(Args&&... args) {
        return {std::make_shared<Impl<T>>(std::forward<Args>(args)...)};
    }

    const std::type_info& get_type_info() const {
        OPENVINO_ASSERT(impl != nullptr, "SharedAny was not initialized");
        return impl->get_type_info();
    }

    template <typename T>
    bool is() const {
        OPENVINO_ASSERT(impl != nullptr, "SharedAny was not initialized");
        return impl->get_type_info() == typeid(T);
    }

    template <typename T>
    T& as() & {
        OPENVINO_ASSERT(is<T>(), "Bad cast from", impl->get_type_info().name(), " to ", typeid(T).name());
        return *static_cast<T*>(impl->get());
    }

    template <typename T>
    const T& as() const& {
        OPENVINO_ASSERT(is<T>(), "Bad cast from", impl->get_type_info().name(), " to ", typeid(T).name());
        return *static_cast<T*>(impl->get());
    }

    template <typename T>
    T&& as() && {
        OPENVINO_ASSERT(is<T>(), "Bad cast from", impl->get_type_info().name(), " to ", typeid(T).name());
        return std::move(*static_cast<T*>(impl->get()));
    }

    template <typename T>
    operator T&() & {
        OPENVINO_ASSERT(is<T>(), "Bad cast from", impl->get_type_info().name(), " to ", typeid(T).name());
        return *static_cast<T*>(impl->get());
    }

    template <typename T>
    operator const T&() const& {
        OPENVINO_ASSERT(is<T>(), "Bad cast from", impl->get_type_info().name(), " to ", typeid(T).name());
        return *static_cast<T*>(impl->get());
    }

    template <typename T>
    operator T &&() && {
        OPENVINO_ASSERT(is<T>(), "Bad cast from", impl->get_type_info().name(), " to ", typeid(T).name());
        return std::move(*static_cast<T*>(impl->get()));
    }

    void print(std::ostream& os) const {
        OPENVINO_ASSERT(impl != nullptr, "SharedAny was not initialized");
        impl->print(os);
    }

    bool operator==(const SharedAny& other) const {
        OPENVINO_ASSERT(impl != nullptr && other.impl != nullptr, "Empty SharedAny is not comparable");
        if (impl == other.impl)
            return true;
        return impl->equal(*other.impl);
    }

    bool operator!=(const SharedAny& other) const {
        return !operator==(other);
    }

    SharedAnyImpl::Ptr get() const {
        return impl;
    }

    operator SharedAnyImpl::Ptr() const {
        return impl;
    }

    std::string to_string() {
        std::stringstream strm;
        impl->print(strm);
        return strm.str();
    }

private:
    SharedAnyImpl::Ptr impl;
};
}  // namespace ov
