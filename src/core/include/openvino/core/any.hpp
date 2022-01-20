// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Any class
 * @file openvino/runtime/any.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace InferenceEngine {
class InferencePlugin;
class ExecutableNetwork;
}  // namespace InferenceEngine

namespace ov {

class Node;
class RuntimeAttribute;
class ParamMap;

namespace runtime {
class CompiledModel;
class RemoteContext;
class RemoteTensor;
class InferencePlugin;
}  // namespace runtime

/**
 * @brief This class represents an object to work with different types
 */
class OPENVINO_API Any {
    std::shared_ptr<void> _so;

    template <typename T>
    using decay_t = typename std::decay<T>::type;

    template <typename T>
    struct IsNullPointer : std::is_same<std::nullptr_t, typename std::remove_cv<T>::type> {};

    template <class T>
    struct Ostreamable {
        template <class U>
        static auto test(U*) -> decltype(std::declval<std::ostream&>() << std::declval<U>(), std::true_type()) {
            return {};
        }
        template <typename>
        static auto test(...) -> std::false_type {
            return {};
        }
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    template <typename T>
    struct EqualityComparable {
        static void* conv(bool);
        template <typename U>
        static char test(decltype(conv(std::declval<U>() == std::declval<U>())));
        template <typename U>
        static long test(...);
        constexpr static const bool value = sizeof(test<T>(nullptr)) == sizeof(char);
    };

    template <typename... T>
    struct EqualityComparable<std::map<T...>> {
        static void* conv(bool);
        template <typename U>
        static char test(decltype(conv(std::declval<typename U::key_type>() == std::declval<typename U::key_type>() &&
                                       std::declval<typename U::mapped_type>() ==
                                           std::declval<typename U::mapped_type>())));
        template <typename U>
        static long test(...);
        constexpr static const bool value = sizeof(test<std::map<T...>>(nullptr)) == sizeof(char);
    };

    template <typename... T>
    struct EqualityComparable<std::vector<T...>> {
        static void* conv(bool);
        template <typename U>
        static char test(decltype(conv(std::declval<typename U::value_type>() ==
                                       std::declval<typename U::value_type>())));
        template <typename U>
        static long test(...);
        constexpr static const bool value = sizeof(test<std::vector<T...>>(nullptr)) == sizeof(char);
    };

    template <typename T>
    struct HasBaseMemberType {
        template <class U>
        static auto test(U*) -> decltype(std::is_class<typename U::Base>::value, std::true_type()) {
            return {};
        }
        template <typename>
        static auto test(...) -> std::false_type {
            return {};
        }
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    template <typename>
    struct TupleToTypeIndex;

    template <typename... Args>
    struct TupleToTypeIndex<std::tuple<Args...>> {
        static std::vector<std::type_index> get() {
            return {typeid(Args)...};
        }
    };

    template <class T>
    struct HasOnAttribute {
        template <class U>
        static auto test(U*)
            -> decltype(std::declval<AttributeVisitor>().on_attribute(std::declval<U&>()), std::true_type()) {
            return {};
        }
        template <typename>
        static auto test(...) -> std::false_type {
            return {};
        }
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    template <class T>
    struct Visitable {
        template <class U>
        static auto test(U*)
            -> decltype(std::declval<U>().visit_attributes(std::declval<AttributeVisitor&>()), std::true_type()) {
            return {};
        }
        template <typename>
        static auto test(...) -> std::false_type {
            return {};
        }
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    /**
     * @brief Base API of erased type
     */
    class OPENVINO_API Base : public std::enable_shared_from_this<Base> {
    public:
        void type_check(const std::type_info&) const;

        using Ptr = std::shared_ptr<Base>;
        virtual const std::type_info& type_info() const = 0;
        virtual std::vector<std::type_index> base_type_info() const = 0;
        virtual const void* addressof() const = 0;
        void* addressof() {
            return const_cast<void*>(const_cast<const Base*>(this)->addressof());
        }
        virtual Base::Ptr copy() const = 0;
        virtual bool equal(const Base& rhs) const = 0;
        virtual void print(std::ostream& os) const = 0;

        virtual const DiscreteTypeInfo& get_type_info() const = 0;
        virtual std::shared_ptr<RuntimeAttribute> as_runtime_attribute() const;
        virtual bool is_copyable() const;
        virtual Any init(const std::shared_ptr<Node>& node);
        virtual Any merge(const std::vector<std::shared_ptr<Node>>& nodes);
        virtual std::string to_string();
        virtual bool visit_attributes(AttributeVisitor&);
        bool visit_attributes(AttributeVisitor&) const;
        std::string to_string() const;

        template <class T>
        bool is() const {
            return typeid(decay_t<T>) == type_info();
        }

        template <class T>
        T& as() & {
            type_check(typeid(decay_t<T>));
            return *static_cast<decay_t<T>*>(addressof());
        }

        template <class T>
        const T& as() const& {
            type_check(typeid(decay_t<T>));
            return *static_cast<const decay_t<T>*>(addressof());
        }

    protected:
        ~Base() = default;
    };

    template <class T, typename = void>
    struct Impl;
    template <class T>
    struct Impl<T, typename std::enable_if<std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value>::type>
        : public Base {
        const DiscreteTypeInfo& get_type_info() const override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->get_type_info();
        }
        std::shared_ptr<RuntimeAttribute> as_runtime_attribute() const override {
            return std::static_pointer_cast<RuntimeAttribute>(runtime_attribute);
        }
        bool is_copyable() const override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->is_copyable();
        }
        Any init(const std::shared_ptr<Node>& node) override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->init(node);
        }
        Any merge(const std::vector<std::shared_ptr<Node>>& nodes) override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->merge(nodes);
        }
        std::string to_string() override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->to_string();
        }

        bool visit_attributes(AttributeVisitor& visitor) override {
            return static_cast<RuntimeAttribute*>(runtime_attribute.get())->visit_attributes(visitor);
        }

        Impl(const T& runtime_attribute) : runtime_attribute{runtime_attribute} {}

        const std::type_info& type_info() const override {
            return typeid(T);
        }

        std::vector<std::type_index> base_type_info() const override {
            return {typeid(std::shared_ptr<RuntimeAttribute>)};
        }

        const void* addressof() const override {
            return std::addressof(runtime_attribute);
        }

        Base::Ptr copy() const override {
            return std::make_shared<Impl<T>>(this->runtime_attribute);
        }

        template <class U>
        static typename std::enable_if<EqualityComparable<U>::value, bool>::type equal_impl(const U& rhs,
                                                                                            const U& lhs) {
            return rhs == lhs;
        }
        template <class U>
        [[noreturn]] static typename std::enable_if<!EqualityComparable<U>::value, bool>::type equal_impl(const U&,
                                                                                                          const U&) {
            throw ov::Exception{"Could not compare types without equality operator"};
        }

        bool equal(const Base& rhs) const override {
            if (rhs.is<T>()) {
                return equal_impl(this->runtime_attribute, rhs.as<T>());
            }
            return false;
        }

        void print(std::ostream& os) const override {
            os << runtime_attribute->to_string();
        }

        T runtime_attribute;
    };

    template <class T>
    struct Impl<T, typename std::enable_if<!std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value>::type>
        : public Base {
        OPENVINO_RTTI(typeid(T).name());

        template <typename... Args>
        Impl(Args&&... args) : value(std::forward<Args>(args)...) {}

        const std::type_info& type_info() const override {
            return typeid(T);
        }

        const void* addressof() const override {
            return std::addressof(value);
        }

        Base::Ptr copy() const override {
            return std::make_shared<Impl<T>>(this->value);
        }

        template <class U>
        static std::vector<std::type_index> base_type_info_impl(
            typename std::enable_if<HasBaseMemberType<U>::value, std::true_type>::type = {}) {
            return TupleToTypeIndex<typename T::Base>::get();
        }
        template <class U>
        static std::vector<std::type_index> base_type_info_impl(
            typename std::enable_if<!HasBaseMemberType<U>::value, std::false_type>::type = {}) {
            return {typeid(T)};
        }

        std::vector<std::type_index> base_type_info() const override {
            return base_type_info_impl<T>();
        }

        template <class U>
        static typename std::enable_if<EqualityComparable<U>::value, bool>::type equal_impl(const U& rhs,
                                                                                            const U& lhs) {
            return rhs == lhs;
        }
        template <class U>
        [[noreturn]] static typename std::enable_if<!EqualityComparable<U>::value, bool>::type equal_impl(const U&,
                                                                                                          const U&) {
            throw ov::Exception{"Could not compare types without equality operator"};
        }

        bool equal(const Base& rhs) const override {
            if (rhs.is<T>()) {
                return equal_impl(this->value, rhs.as<T>());
            }
            return false;
        }

        template <class U>
        static typename std::enable_if<Ostreamable<U>::value>::type print_impl(std::ostream& os, const U& value) {
            os << value;
        }

        template <class U>
        static typename std::enable_if<!Ostreamable<U>::value>::type print_impl(std::ostream&, const U&) {}

        void print(std::ostream& os) const override {
            print_impl(os, value);
        }

        T value;
    };

    friend class ::ov::RuntimeAttribute;
    friend class ::ov::ParamMap;
    friend class ::InferenceEngine::InferencePlugin;
    friend class ::InferenceEngine::ExecutableNetwork;
    friend class ::ov::runtime::CompiledModel;
    friend class ::ov::runtime::RemoteContext;
    friend class ::ov::runtime::RemoteTensor;
    friend class ::ov::runtime::InferencePlugin;

    Any(const Any& other, const std::shared_ptr<void>& so);

    void impl_check() const;

    mutable Base::Ptr _runtime_attribute_impl;

    Base::Ptr _impl;

public:
    /// @brief Default constructor
    Any() = default;

    /// @brief Default copy constructor
    /// @param other other Any object
    Any(const Any& other) = default;

    /// @brief Default copy assignment operator
    /// @param other other Any object
    /// @return reference to the current object
    Any& operator=(const Any& other) = default;

    /// @brief Default move constructor
    /// @param other other Any object
    Any(Any&& other) = default;

    /// @brief Default move assignment operator
    /// @param other other Any object
    /// @return reference to the current object
    Any& operator=(Any&& other) = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~Any();

    /**
     * @brief Constructor creates any with object
     *
     * @tparam T Any type
     * @param value object
     */
    template <typename T,
              typename std::enable_if<!std::is_same<decay_t<T>, Any>::value && !std::is_abstract<decay_t<T>>::value &&
                                          !std::is_convertible<decay_t<T>, Base::Ptr>::value,
                                      bool>::type = true>
    Any(T&& value) : _impl{std::make_shared<Impl<decay_t<T>>>(std::forward<T>(value))} {}

    /**
     * @brief Constructor creates string any from char *
     *
     * @param str char array
     */
    Any(const char* str);

    /**
     * @brief Empty constructor
     *
     */
    Any(const std::nullptr_t);

    /**
     * @brief Inplace value construction function
     *
     * @tparam T Any type
     * @tparam Args pack of paramter types passed to T constructor
     * @param args pack of paramters passed to T constructor
     */
    template <typename T, typename... Args>
    static Any make(Args&&... args) {
        Any any;
        any._impl = std::make_shared<Impl<decay_t<T>>>(std::forward<Args>(args)...);
        return any;
    }

    /**
     * Returns type info
     * @return type info
     */
    const std::type_info& type_info() const;

    /**
     * Checks that any contains a value
     * @return false if any contains a value else false
     */
    bool empty() const;

    /**
     * @brief check the type of value in any
     * @tparam T Type of value
     * @return true if type of value is correct
     */
    template <class T>
    bool is() const {
        if (_impl != nullptr) {
            if (_impl->type_info() == typeid(decay_t<T>)) {
                return true;
            }
            for (const auto& type_index : _impl->base_type_info()) {
                if (type_index == typeid(decay_t<T>)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <typename T>
    typename std::enable_if<std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value, T>::type&& as() && {
        if (_impl == nullptr) {
            _runtime_attribute_impl = std::make_shared<Impl<decay_t<T>>>(T{});
            return _runtime_attribute_impl->as<T>();
        } else {
            if (_impl->type_info() == typeid(decay_t<T>)) {
                return std::move(*static_cast<decay_t<T>*>(_impl->addressof()));
            } else {
                auto runtime_attribute = _impl->as_runtime_attribute();
                if (runtime_attribute == nullptr) {
                    throw ov::Exception{
                        std::string{"Any does not contains pointer to runtime_attribute. It contains "} +
                        _impl->type_info().name()};
                }
                auto vptr = std::dynamic_pointer_cast<typename T::element_type>(runtime_attribute);
                if (vptr == nullptr && T::element_type::get_type_info_static() != runtime_attribute->get_type_info() &&
                    T::element_type::get_type_info_static() != RuntimeAttribute::get_type_info_static()) {
                    throw ov::Exception{std::string{"Could not cast Any runtime_attribute to "} + typeid(T).name() +
                                        " from " + _impl->type_info().name() + "; from " +
                                        static_cast<std::string>(runtime_attribute->get_type_info()) + " to " +
                                        static_cast<std::string>(T::element_type::get_type_info_static())};
                }
                vptr = std::static_pointer_cast<typename T::element_type>(runtime_attribute);
                _runtime_attribute_impl = std::make_shared<Impl<decay_t<T>>>(vptr);
                return _runtime_attribute_impl->as<T>();
            }
        }
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    typename std::enable_if<std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value, T>::type& as() & {
        if (_impl == nullptr) {
            _runtime_attribute_impl = std::make_shared<Impl<decay_t<T>>>(T{});
            return _runtime_attribute_impl->as<T>();
        } else {
            if (_impl->type_info() == typeid(decay_t<T>)) {
                return *static_cast<decay_t<T>*>(_impl->addressof());
            } else {
                auto runtime_attribute = _impl->as_runtime_attribute();
                if (runtime_attribute == nullptr) {
                    throw ov::Exception{
                        std::string{"Any does not contains pointer to runtime_attribute. It contains "} +
                        _impl->type_info().name()};
                }
                auto vptr = std::dynamic_pointer_cast<typename T::element_type>(runtime_attribute);
                if (vptr == nullptr && T::element_type::get_type_info_static() != runtime_attribute->get_type_info() &&
                    T::element_type::get_type_info_static() != RuntimeAttribute::get_type_info_static()) {
                    throw ov::Exception{std::string{"Could not cast Any runtime_attribute to "} + typeid(T).name() +
                                        " from " + _impl->type_info().name() + "; from " +
                                        static_cast<std::string>(runtime_attribute->get_type_info()) + " to " +
                                        static_cast<std::string>(T::element_type::get_type_info_static())};
                }
                vptr = std::static_pointer_cast<typename T::element_type>(runtime_attribute);
                _runtime_attribute_impl = std::make_shared<Impl<decay_t<T>>>(vptr);
                return _runtime_attribute_impl->as<T>();
            }
        }
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    const typename std::enable_if<std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value, T>::type& as()
        const& {
        if (_impl == nullptr) {
            _runtime_attribute_impl = std::make_shared<Impl<decay_t<T>>>(T{});
            return _runtime_attribute_impl->as<T>();
        } else {
            if (_impl->type_info() == typeid(decay_t<T>)) {
                return *static_cast<const decay_t<T>*>(_impl->addressof());
            } else {
                auto runtime_attribute = _impl->as_runtime_attribute();
                if (runtime_attribute == nullptr) {
                    throw ov::Exception{
                        std::string{"Any does not contains pointer to runtime_attribute. It contains "} +
                        _impl->type_info().name()};
                }
                auto vptr = std::dynamic_pointer_cast<typename T::element_type>(runtime_attribute);
                if (vptr == nullptr && T::element_type::get_type_info_static() != runtime_attribute->get_type_info() &&
                    T::element_type::get_type_info_static() != RuntimeAttribute::get_type_info_static()) {
                    throw ov::Exception{std::string{"Could not cast Any runtime_attribute to "} + typeid(T).name() +
                                        " from " + _impl->type_info().name() + "; from " +
                                        static_cast<std::string>(runtime_attribute->get_type_info()) + " to " +
                                        static_cast<std::string>(T::element_type::get_type_info_static())};
                }
                vptr = std::static_pointer_cast<typename T::element_type>(runtime_attribute);
                _runtime_attribute_impl = std::make_shared<Impl<decay_t<T>>>(vptr);
                return _runtime_attribute_impl->as<T>();
            }
        }
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <typename T>
    typename std::enable_if<!std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value, T>::type&& as() && {
        impl_check();
        _impl->type_check(typeid(decay_t<T>));
        return std::move(*static_cast<decay_t<T>*>(_impl->addressof()));
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    typename std::enable_if<!std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value, T>::type& as() & {
        impl_check();
        if (_impl->type_info() == typeid(decay_t<T>)) {
            return *static_cast<decay_t<T>*>(_impl->addressof());
        }
        for (const auto& type_index : _impl->base_type_info()) {
            if (type_index == typeid(decay_t<T>)) {
                return *static_cast<decay_t<T>*>(_impl->addressof());
            }
        }
        throw ov::Exception{std::string{"Bad cast from: "} + _impl->type_info().name() + " to: " + typeid(T).name()};
    }

    /**
     * Dynamic cast to specified type
     * @tparam T type
     * @return casted object
     */
    template <class T>
    const typename std::enable_if<!std::is_convertible<T, std::shared_ptr<RuntimeAttribute>>::value, T>::type& as()
        const& {
        impl_check();
        if (_impl->type_info() == typeid(decay_t<T>)) {
            return *static_cast<const decay_t<T>*>(_impl->addressof());
        }
        for (const auto& type_index : _impl->base_type_info()) {
            if (type_index == typeid(decay_t<T>)) {
                return *static_cast<const decay_t<T>*>(_impl->addressof());
            }
        }
        throw ov::Exception{std::string{"Bad cast from: "} + _impl->type_info().name() + " to: " + typeid(T).name()};
    }

    /**
     * @brief Converts to specified type
     * @tparam T type
     * @return casted object
     */
    template <typename T>
    operator T&() & {
        return as<T>();
    }

    /**
     * @brief Converts to specified type
     * @tparam T type
     * @return casted object
     */
    template <typename T>
    operator const T&() const& {
        return as<T>();
    }

    /**
     * @brief Converts to specified type
     * @tparam T type
     * @return casted object
     */
    template <typename T>
    operator T&() const& {
        return const_cast<Any*>(this)->as<T>();
    }

    /**
     * @brief Converts to specified type
     * @tparam T type
     * @return casted object
     */
    template <typename T>
    operator T &&() && {
        return std::move(as<T&&>());
    }

    /**
     * @brief The comparison operator for the Any
     *
     * @param other object to compare
     * @return true if objects are equal
     */
    bool operator==(const Any& other) const;

    /**
     * @brief The comparison operator for the Any
     *
     * @param other object to compare
     * @return true if objects are equal
     */
    bool operator==(const std::nullptr_t&) const;

    /**
     * @brief The comparison operator for the Any
     *
     * @param other object to compare
     * @return true if objects aren't equal
     */
    bool operator!=(const Any& other) const;

    /**
     * @brief Standard pointer operator
     * @return underlined interface
     */
    OPENVINO_DEPRECATED("Please use as() method")
    Base* operator->();

    /**
     * @brief Standard pointer operator
     * @return underlined interface
     */
    OPENVINO_DEPRECATED("Please use as() method")
    const Base* operator->() const;

    /**
     * @brief Prints underlying object to the given output stream.
     * Uses operator<< if it is defined, leaves stream unchanged otherwise.
     * In case of empty any or nullptr stream immediately returns.
     *
     * @param stream Output stream object will be printed to.
     */
    void print(std::ostream& stream) const;

    /**
     * @brief Return pointer to underlined interface
     * @return underlined interface
     */
    OPENVINO_DEPRECATED("Please use as() method")
    Base* get() {
        impl_check();
        return _impl.get();
    }

    /**
     * @brief Return pointer to underlined interface
     * @return underlined interface
     */
    OPENVINO_DEPRECATED("Please use as() method")
    const Base* get() const {
        impl_check();
        return _impl.get();
    }
};

namespace util {
template <>
struct AsTypePtr<Any> {
    template <typename T>
    OPENVINO_DEPRECATED("Please use ov::Any::as() method")
    static std::shared_ptr<T> call(const Any& any) {
        try {
            return any.as<std::shared_ptr<T>>();
        } catch (...) {
            return {};
        }
    }
};
}  // namespace util

using RTMap = std::map<std::string, Any>;

using AnyVector = std::vector<ov::Any>;

}  // namespace ov

namespace std {
template <typename T>
OPENVINO_DEPRECATED("Please use ov::Any::as() method")
std::shared_ptr<T> dynamic_pointer_cast(const ::ov::Any& any) {
    try {
        return any.as<std::shared_ptr<T>>();
    } catch (...) {
        return {};
    }
}

template <typename T>
OPENVINO_DEPRECATED("Please use ov::Any::as() method")
std::shared_ptr<T> static_pointer_cast(const ::ov::Any& any) {
    return any.as<std::shared_ptr<T>>();
}

}  // namespace std
