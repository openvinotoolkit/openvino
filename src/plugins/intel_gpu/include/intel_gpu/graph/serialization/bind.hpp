// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <functional>
#include "buffer.hpp"
#include "static_instance.hpp"

#define DECLARE_OBJECT_TYPE_SERIALIZATION(cls_name)                        \
    static const std::string& get_type_info_s() {                               \
        static const std::string type_name = #cls_name;                    \
        return type_name;                                                  \
    }                                                                      \
    const std::string& get_type_info() const override {  return get_type_info_s(); }

#define BIND_TO_BUFFER(buffer, type)                                                       \
        template <>                                                                        \
        class bind_creator<buffer, type> {                                                 \
        private:                                                                           \
            static const instance_creator<buffer, type>& creator;                          \
        };                                                                                 \
        const instance_creator<buffer, type>& bind_creator<buffer, type>::creator =        \
            static_instance<instance_creator<buffer, type>>::get_instance().instantiate();

namespace cldnn {

template <typename BufferType>
struct saver_storage {
    using save_function = std::function<void(BufferType&, const void*)>;
    using value_type = typename std::unordered_map<std::string, save_function>::value_type;

    static saver_storage<BufferType>& instance() {
        static saver_storage<BufferType> instance;
        return instance;
    }

    const save_function& get_save_function(const std::string& type) const {
        return map.at(type);
    }

    void set_save_function(const value_type& pair) {
        map.insert(pair);
    }

private:
    saver_storage() = default;
    saver_storage(const saver_storage&) = delete;
    void operator=(const saver_storage&) = delete;

    std::unordered_map<std::string, save_function> map;
};

template <typename T>
struct void_deleter {
    void operator()(const T*) const { }
};

template <typename BufferType, typename FuncT>
struct loader_storage {
    using value_type = typename std::unordered_map<std::string, FuncT>::value_type;

    static loader_storage& instance() {
        static loader_storage instance;
        return instance;
    }

    const FuncT& get_load_function(const std::string& type) {
        return map.at(type);
    }

    void set_load_function(const value_type& pair) {
        map.insert(pair);
    }

private:
    loader_storage() = default;
    loader_storage(const loader_storage&) = delete;
    void operator=(const loader_storage&) = delete;

    std::unordered_map<std::string, FuncT> map;
};

template <typename BufferType>
using def = loader_storage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, void_deleter<void>>&)>>;

template <typename BufferType>
using dif = loader_storage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, void_deleter<void>>&, engine&)>>;

template <typename BufferType, typename T, typename Enable = void>
class buffer_binder;

template <typename BufferType, typename T>
class buffer_binder<BufferType, T, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static buffer_binder& instance() {
        static buffer_binder instance;
        return instance;
    }

private:
    buffer_binder() {
        saver_storage<BufferType>::instance().set_save_function({T::get_type_info_s(), save});
    }

    buffer_binder(const buffer_binder&) = delete;
    void operator=(const buffer_binder&) = delete;

    template <typename Derived>
    static const Derived* downcast(const void* base_ptr) {
        return static_cast<Derived const *>(base_ptr);
    }

    static void save(BufferType& buffer, const void* base_ptr) {
        const auto derived_ptr = downcast<T>(base_ptr);
        derived_ptr->save(buffer);
    }
};

template <typename BufferType, typename T>
class buffer_binder<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                           std::is_default_constructible<T>::value>::type> {
public:
    static buffer_binder& instance() {
        static buffer_binder instance;
        return instance;
    }

private:
    buffer_binder() {
        def<BufferType>::instance().set_load_function(
                {T::get_type_info_s(), [](BufferType& buffer, std::unique_ptr<void, void_deleter<void>>& result_ptr) {
            std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T());
            derived_ptr->load(buffer);
            result_ptr.reset(derived_ptr.release());
        }});
    }

    buffer_binder(const buffer_binder&) = delete;
    void operator=(const buffer_binder&) = delete;
};

template <typename BufferType, typename T>
class buffer_binder<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                           !std::is_default_constructible<T>::value>::type> {
public:
    static buffer_binder& instance() {
        static buffer_binder instance;
        return instance;
    }

private:
    buffer_binder() {
        dif<BufferType>::instance().set_load_function(
                {T::get_type_info_s(), [](BufferType& buffer, std::unique_ptr<void, void_deleter<void>>& result_ptr, engine& engine) {
            std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(engine));
            derived_ptr->load(buffer);
            result_ptr.reset(derived_ptr.release());
        }});
    }

    buffer_binder(const buffer_binder&) = delete;
    void operator=(const buffer_binder&) = delete;
};

template <typename BufferType, typename T>
class bind_creator;

template <typename BufferType, typename T>
class instance_creator {
public:
    const instance_creator& instantiate() {
        static_instance<buffer_binder<BufferType, T>>::get_instance();
        return *this;
    }
};

}  // namespace cldnn
