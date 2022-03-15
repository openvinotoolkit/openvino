// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains the class that provides access to runtime object accesses
 * @file accesses.hpp
 */
#pragma once

#include <functional>
#include <string>
#include <type_traits>

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @brief This class to configure access to accesses of runtime objects
 */
class INFERENCE_ENGINE_API_CLASS(PropertyAccess) {
    template <typename T>
    struct Identity {
        using type = T;
    };
    struct SubAccess;
    struct Access {
        using Ptr = std::shared_ptr<Access>;
        virtual Any get(const AnyMap&) const = 0;
        virtual void set(const Any&){};
        virtual void precondition(const Any&) const {};
        virtual bool is_mutable() const {
            return false;
        };
        virtual SubAccess* sub_access_ptr() {
            return nullptr;
        }
        const SubAccess* sub_access_ptr() const {
            return const_cast<Access*>(this)->sub_access_ptr();
        }
        bool is_sub_access() const {
            return sub_access_ptr() != nullptr;
        }
        SubAccess& sub_access() {
            OPENVINO_ASSERT(is_sub_access(), "Not property sub access");
            return *sub_access_ptr();
        }
        const SubAccess& sub_access() const {
            OPENVINO_ASSERT(is_sub_access(), "Not property sub access");
            return *sub_access_ptr();
        }
        virtual void ro(){};

    protected:
        ~Access() = default;
    };

    using AccessMap = std::map<std::string, std::shared_ptr<Access>>;
    template <class T>
    struct IsGetter {
        template <class U>
        static auto test(U*) -> decltype(std::declval<U>()(), std::true_type()) {
            return {};
        }
        template <class U>
        static auto test(U*) -> decltype(std::declval<U>()(std::declval<const AnyMap&>()), std::true_type()) {
            return {};
        }
        template <typename>
        static auto test(...) -> std::false_type {
            return {};
        }
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    template <class T>
    struct IsGetterArg {
        template <class U>
        static auto test(U*) -> decltype(std::declval<U>()(std::declval<const AnyMap&>()), std::true_type()) {
            return {};
        }
        template <typename>
        static auto test(...) -> std::false_type {
            return {};
        }
        constexpr static const auto value = std::is_same<std::true_type, decltype(test<T>(nullptr))>::value;
    };

    template <class T>
    using IsProperties = std::is_same<PropertyAccess, typename std::decay<T>::type>;

    template <typename>
    struct IsRef : public std::false_type {};

    template <class T>
    struct IsRef<std::reference_wrapper<T>> : public std::true_type {};
    struct SetterArg {
        template <typename T>
        operator const T&() const {
            return any.as<T>();
        }
        template <typename T>
        operator const T() const {
            return any.as<T>();
        }
        operator const Any&() const {
            return any;
        }
        const Any& any;
    };

    template <typename T>
    using AsAnyType = typename std::conditional<
        (std::is_array<T>::value &&
         std::is_same<typename std::decay<typename std::remove_all_extents<T>::type>::type, char>::value) ||
            std::is_same<typename std::decay<T>::type, const char*>::value,
        std::string,
        T>::type;

    std::vector<std::vector<std::string>> get_all_pathes() const;

    std::vector<std::vector<std::string>> find_property(const std::vector<std::string>& rout) const;

    std::vector<PropertyName> get_supported() const;

    const void* find_access(const std::vector<std::string>& path) const;
    void* find_access(const std::vector<std::string>& path);

public:
    /**
     * @brief Default constructor
     */
    PropertyAccess();

    /**
     * @brief Set root property name
     * @param name_ root property name
     * @return Reference to current object
     */
    PropertyAccess& set_name(const std::string& name_) {
        name = name_;
        return *this;
    }

private:
    template <typename...>
    struct FunctionAccess;
    template <typename G>
    struct FunctionAccess<G> : public Access {
        template <typename Get>
        static typename std::enable_if<!IsGetterArg<Get>::value, Any>::type call_get(Get get, const AnyMap&) {
            return get();
        }
        template <typename Get>
        static typename std::enable_if<IsGetterArg<Get>::value, Any>::type call_get(Get get, const AnyMap& args) {
            return get(args);
        }
        FunctionAccess(G get) : get_impl{std::move(get)} {}
        Any get(const AnyMap& args) const override {
            return call_get(get_impl, args);
        }
        bool is_mutable() const override {
            return false;
        }
        G get_impl;
    };

    template <typename G, typename S>
    struct FunctionAccess<G, S> : public FunctionAccess<G> {
        FunctionAccess(G get, S set) : FunctionAccess<G>{std::move(get)}, set_impl{std::move(set)} {}
        void set(const Any& any) override {
            set_impl(SetterArg{any});
        }
        bool is_mutable() const override {
            return mutability == PropertyMutability::RW;
        }
        void ro() override {
            mutability = PropertyMutability::RO;
        }
        S set_impl;
        PropertyMutability mutability = PropertyMutability::RW;
    };

    template <typename G, typename S, typename P>
    struct FunctionAccess<G, S, P> : public FunctionAccess<G, S> {
        FunctionAccess(G get, S set, P precondition)
            : FunctionAccess<G, S>{std::move(get), std::move(set)},
              precondition_impl{std::move(precondition)} {}
        void precondition(const Any& any) const override {
            precondition_impl(SetterArg{any});
        }
        P precondition_impl;
    };

public:
    /**
     * @brief Add readwrite property access
     * @tparam G property getter type
     * @tparam S property setter type
     * @param name property name
     * @param get property getter
     * @param set property setter
     * @return Reference to current object
     */
    template <typename G, typename... Args>
    typename std::enable_if<IsGetter<G>::value, PropertyAccess>::type& add(const std::string& name,
                                                                           G get,
                                                                           Args&&... args) {
        accesses[name] = std::make_shared<FunctionAccess<G, Args...>>(std::move(get), std::move(args)...);
        return *this;
    }

private:
    template <typename...>
    struct Value;

    template <typename T>
    struct Value<T> : public Access {
        Value(const T& default_value, const PropertyMutability mutability_)
            : value{default_value},
              mutability{mutability_} {}
        Any get(const AnyMap&) const override {
            return value;
        }
        void set(const Any& any) override {
            value = any.as<AsAnyType<T>>();
        }
        bool is_mutable() const override {
            return mutability == PropertyMutability::RW;
        }
        void ro() override {
            mutability = PropertyMutability::RO;
        }
        AsAnyType<T> value;
        PropertyMutability mutability = PropertyMutability::RW;
    };
    template <typename T, typename P>
    struct Value<T, P> : public Value<T> {
        Value(const T& default_value, const P precondition_)
            : Value<T>{default_value, PropertyMutability::RW},
              precondition_impl{std::move(precondition_)} {}
        void precondition(const Any& any) const override {
            precondition_impl(SetterArg{any});
        }
        P precondition_impl;
    };

public:
    /**
     * @brief Add read only property access
     * @tparam T property value type
     * @param name property name
     * @param deafault_value default property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T>
    typename std::enable_if<!IsGetter<T>::value && !IsProperties<T>::value && !IsRef<T>::value, PropertyAccess>::type&
    add(const std::string& name,
        const T& deafault_value,
        const PropertyMutability mutability = PropertyMutability::RW) {
        accesses[name] = std::make_shared<Value<T>>(deafault_value, mutability);
        return *this;
    }

    /**
     * @brief Add read only property access
     * @tparam T property value type
     * @param name property name
     * @param deafault_value default property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T, typename P>
    typename std::enable_if<!IsGetter<T>::value && !IsProperties<T>::value && !IsRef<T>::value, PropertyAccess>::type&
    add(const std::string& name, const T& deafault_value, P precondition) {
        accesses[name] = std::make_shared<Value<T, P>>(deafault_value, std::move(precondition));
        return *this;
    }

private:
    template <typename...>
    struct Ref;
    template <typename T, typename R>
    struct Ref<T, R> : public Access {
        Ref(std::reference_wrapper<R> ref_, const PropertyMutability mutability_)
            : ref{ref_},
              mutability{mutability_} {}
        Any get(const AnyMap& args) const override {
            return T(ref.get());
        }
        void set(const Any& any) override {
            ref.get() = any.as<AsAnyType<T>>();
        }
        bool is_mutable() const override {
            return mutability == PropertyMutability::RW;
        }
        void ro() override {
            mutability = PropertyMutability::RO;
        }
        std::reference_wrapper<R> ref;
        PropertyMutability mutability = PropertyMutability::RW;
    };
    template <typename T, typename R, typename P>
    struct Ref<T, R, P> : public Ref<T, R> {
        Ref(std::reference_wrapper<R> ref_, const P precondition_)
            : Ref<T, R>{ref_, PropertyMutability::RW},
              precondition_impl{std::move(precondition_)} {}
        void precondition(const Any& any) const override {
            precondition_impl(SetterArg{any});
        }
        P precondition_impl;
    };

public:
    /**
     * @brief Add property access
     * @tparam T property value type
     * @param name property name
     * @param ref reference wrapper to property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T>
    PropertyAccess& add(const std::string& name,
                        std::reference_wrapper<T> ref,
                        const PropertyMutability mutability = PropertyMutability::RW) {
        accesses[name] = std::make_shared<Ref<T, T>>(ref, mutability);
        return *this;
    }

    /**
     * @brief Add property access
     * @tparam T property value type
     * @param name property name
     * @param ref reference wrapper to property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T, typename P>
    PropertyAccess& add(const std::string& name, std::reference_wrapper<T> ref, P precondition) {
        accesses[name] = std::make_shared<Ref<T, T, P>>(ref, std::move(precondition));
        return *this;
    }

    /**
     * @brief Add read only property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property property variable
     * @param ref reference wrapper to property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T, PropertyMutability M>
    PropertyAccess& add(const Property<T, M>& property, const typename Identity<T>::type& default_value = {}) {
        return add(property.name(), default_value, M);
    }

    /**
     * @brief Add read only property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property property variable
     * @param ref reference wrapper to property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T, typename P>
    PropertyAccess& add(const Property<T, PropertyMutability::RW>& property,
                        const typename Identity<T>::type& default_value,
                        P precondition) {
        return add(property.name(), default_value, std::move(precondition));
    }

    /**
     * @brief Add read only property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property property variable
     * @param ref reference wrapper to property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T, PropertyMutability M, typename R>
    PropertyAccess& add(const Property<T, M>& property, std::reference_wrapper<R> ref) {
        accesses[property.name()] = std::make_shared<Ref<T, R>>(ref, M);
        return *this;
    }

    /**
     * @brief Add read only property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property property variable
     * @param ref reference wrapper to property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T, typename P, typename R>
    PropertyAccess& add(const Property<T, PropertyMutability::RW>& property,
                        std::reference_wrapper<R> ref,
                        P precondition) {
        accesses[property.name()] = std::make_shared<Ref<T, R, P>>(ref, std::move(precondition));
        return *this;
    }

    /**
     * @brief Add read only property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property property variable
     * @param ref reference wrapper to property value
     * @param precondition set property precondition
     * @return Reference to current object
     */
    template <typename T, PropertyMutability M, typename G, typename... Args>
    typename std::enable_if<IsGetter<G>::value, PropertyAccess>::type& add(const Property<T, M>& property,
                                                                           G get,
                                                                           Args&&... args) {
        if (M == PropertyMutability::RW) {
            return add(property.name(), std::move(get), std::forward<Args>(args)...);
        } else {
            return add(property.name(), std::move(get), std::forward<Args>(args)...).ro(property);
        }
    }

    /**
     * @brief Add content of property access
     * @param sub_accesses sub object set of properties
     * @param mutability sub properties mutability
     * @return Reference to current object
     */
    PropertyAccess& add(PropertyAccess sub_accesses);

    /**
     * @brief Add property access for readwrite property
     * @param name prefix name
     * @param sub_accesses sub object set of properties
     * @param mutability sub properties mutability
     * @return Reference to current object
     */
    PropertyAccess& add(const std::string& name, PropertyAccess sub_accesses);

    /**
     * @brief Add read only property access for readwrite property
     * @param name property set object name
     * @param sub_accesses sub object set of properties
     * @param mutability sub properties mutability
     * @return Reference to current object
     */
    PropertyAccess& add(const NamedProperties& named_properties, PropertyAccess sub_accesses);

    /**
     * @brief Remove property access using defined name
     * @param name property name
     * @return Reference to current object
     */
    PropertyAccess& remove(const std::string& name);

    /**
     * @brief remove property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property variable
     * @return Reference to current object
     */
    template <typename T, PropertyMutability M>
    PropertyAccess& remove(const Property<T, M>& property) {
        return remove(property.name());
    }

    /**
     * @brief make all properties read only
     * @return Reference to current object
     */
    PropertyAccess& ro();

    /**
     * @brief make property read only
     * @param name property name
     * @return Reference to current object
     */
    PropertyAccess& ro(const std::string& name);

    /**
     * @brief remove property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property variable
     * @return Reference to current object
     */
    template <typename T, PropertyMutability M>
    PropertyAccess& ro(const Property<T, M>& property) {
        return ro(property.name());
    }

    /**
     * @brief Get property.
     * @param name property name
     * @return property value if exist, empty any otherwise
     */
    Any find(const std::string name, const AnyMap& args = {}) const;

    /**
     * @brief check if property with name was added
     * @param name property name
     * @return true if property exist
     */
    bool has(const std::string name) const;

    /**
     * @brief remove property access
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property variable
     * @return true if property exist
     */
    template <typename T, PropertyMutability M>
    bool has(const Property<T, M>& property) {
        return has(property.name());
    }

    /**
     * @brief Returns all properties values
     * @param mutability optional parameter that define what type of properties will be returned
     * PropertyMutability::RO - ALl readable values will be returned
     * PropertyMutability::RW - Only mutable values will be returned
     * @return map of property values wrapped into Any
     */
    AnyMap get(const PropertyMutability mutability = PropertyMutability::RW) const;

    /**
     * @brief Returns the value bound to name
     * @param name property name
     * @return Any with property value
     */
    Any get(const std::string& name, const AnyMap& args = {}) const;

    /**
     * @brief Returns the value bound to property object
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property variable
     * @return Any with property value
     */
    template <typename T, PropertyMutability M>
    T get(const Property<T, M>& property, const AnyMap& args = {}) const {
        return get(property.name(), args).template as<T>();
    }

    /**
     * @brief Returns the value bound to property object
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property variable
     * @return Any with property value
     */
    template <typename T, PropertyMutability M>
    T get(const Property<T, M>& property, const AnyMap& args = {}) {
        return get(property.name(), args).template as<T>();
    }

    /**
     * @brief Set property value that is bound to name
     * @param name property name
     * @param property property value
     */
    PropertyAccess& set(const std::string& name, const Any& property);

    /**
     * @brief Set property value that is bound to name
     * @tparam T property value type
     * @tparam M property mutability
     * @param property property variable
     * @param value property value
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<PropertyAccess&, Properties...> set(Properties&&... properties) {
        return set(AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Set property values that is bound to names
     * @param properties map with property values
     */
    PropertyAccess& set(const AnyMap& properties);

    /**
     * @brief Set property values that are defined by the string
     * @param properties map with property strings
     */
    PropertyAccess& set(const std::map<std::string, std::string>& properties);

    /**
     * @brief Merge given property values with current values. If property value is presented inside
     * it will be overwritten by other property value
     * @param other map with property strings
     * @param mutability optional parameter that define what type of properties will be merged
     * PropertyMutability::RO - ALl readable values will be merged
     * PropertyMutability::RW - Only mutable values will be merged
     * @return map of property values wrapped into Any
     */
    AnyMap merge(const AnyMap& other, const PropertyMutability mutability = PropertyMutability::RW) const;

    /**
     * @brief Merge given property values with current values. If property value is presented inside
     * it will be overwritten by other property value
     * @param other map with property strings
     * @param mutability optional parameter that define what type of properties will be merged
     * PropertyMutability::RO - ALl readable values will be merged
     * PropertyMutability::RW - Only mutable values will be merged
     * @return map of property values represented as strings
     */
    std::map<std::string, std::string> merge(const std::map<std::string, std::string>& other,
                                             const PropertyMutability mutability = PropertyMutability::RW) const;

    /**
     * @brief Returns whether there is no added properties
     * @return true if there is no added priporties
     */
    bool empty() const;

private:
    std::string name;
    AccessMap accesses;
};
}  // namespace ov