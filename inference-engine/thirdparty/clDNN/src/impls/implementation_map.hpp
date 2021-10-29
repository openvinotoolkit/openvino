// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <functional>
#include <typeinfo>
#include <tuple>
#include <string>

namespace cldnn {

template <typename T, typename U>
class singleton_map : public std::map<T, U> {
    singleton_map() : std::map<T, U>() {}
    singleton_map(singleton_map const&) = delete;
    void operator=(singleton_map const&) = delete;

public:
    static singleton_map& instance() {
        static singleton_map instance_;
        return instance_;
    }
};


struct permute;
struct reorder;
struct custom_gpu_primitive;
struct generic_layer;
struct reshape;
struct data;
struct mutable_data;
struct input_layout;
struct prior_box;
struct loop;

struct primitive_impl;

template <class PType>
struct typed_program_node;

template <typename primitive_kind>
struct implementation_key {
    typedef std::tuple<data_types, format::type> type;
    type operator()(const typed_program_node<primitive_kind>& primitive) {
        return std::make_tuple(primitive.get_dependency(0).get_output_layout().data_type,
                               primitive.get_dependency(0).get_output_layout().format);
    }
    type operator()(const layout& proposed_layout) {
        return std::make_tuple(proposed_layout.data_type, proposed_layout.format);
    }
};

template <>
struct implementation_key<permute> {
    typedef int32_t type;
    type operator()(const typed_program_node<permute>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<reorder> {
    typedef int32_t type;
    type operator()(const typed_program_node<reorder>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<generic_layer> {
    typedef int32_t type;
    type operator()(const typed_program_node<generic_layer>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<custom_gpu_primitive> {
    typedef int32_t type;
    type operator()(const typed_program_node<custom_gpu_primitive>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<reshape> {
    typedef int32_t type;
    type operator()(const typed_program_node<reshape>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<data> {
    typedef int32_t type;
    type operator()(const typed_program_node<data>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<mutable_data> {
    typedef int32_t type;
    type operator()(const typed_program_node<mutable_data>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<input_layout> {
    typedef int32_t type;
    type operator()(const typed_program_node<input_layout>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<prior_box> {
    typedef int32_t type;
    type operator()(const typed_program_node<prior_box>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <>
struct implementation_key<loop> {
    typedef int32_t type;
    type operator()(const typed_program_node<loop>&) { return -1; }
    type operator()(const layout&) { return -1; }
};

template <typename primitive_kind>
class implementation_map {
public:
    using key_builder = implementation_key<primitive_kind>;
    using key_type = typename key_builder::type;
    using factory_type = std::function<primitive_impl*(const typed_program_node<primitive_kind>&)>;
    using map_type = singleton_map<impl_types, std::pair<std::set<key_type>, factory_type>>;

    static factory_type get(const typed_program_node<primitive_kind>& primitive) {
        impl_types target_impl_type = primitive.get_preferred_impl_type();
        // lookup in database; throw if not found
        auto key = key_builder()(primitive);
        for (auto& kv : map_type::instance()) {
            impl_types impl_type = kv.first;
            if ((target_impl_type & impl_type) != impl_type)
                continue;

            std::set<key_type>& keys_set = kv.second.first;
            auto& factory = kv.second.second;
            if (keys_set.empty() || keys_set.find(key) != keys_set.end()) {
                return factory;
            }
        }
        throw std::runtime_error(std::string("implementation_map for ") + typeid(primitive_kind).name() +
                                     " could not find any implementation to match key");
    }

    // check if for a given engine and type there exist an implementation
    static bool check(const typed_program_node<primitive_kind>& primitive) {
        impl_types target_impl_type = primitive.get_preferred_impl_type();
        auto key = key_builder()(primitive);
        return check_key(target_impl_type, key);
    }

    // check if there exists a kernel implementation of a primitive with output set it primitive's output layout
    static bool check_io_eq(const typed_program_node<primitive_kind>& primitive) {
        impl_types target_impl_type = primitive.get_preferred_impl_type();
        auto key = key_builder()(primitive.get_output_layout());
        return check_key(target_impl_type, key);
    }

    static bool check_key(impl_types target_impl_type, key_type key) {
        for (auto& kv : map_type::instance()) {
            impl_types impl_type = kv.first;
            if ((target_impl_type & impl_type) != impl_type)
                continue;
            std::set<key_type>& keys_set = kv.second.first;
            if (keys_set.empty())
                return true;
            return keys_set.find(key) != keys_set.end();
        }
        return false;
    }

    static void add(impl_types impl_type, factory_type factory, std::set<key_type> keys) {
        if (impl_type == impl_types::any) {
            throw std::runtime_error("[CLDNN] Can't register impl with type any");
        }
        map_type::instance().insert({impl_type, {keys, factory}});
    }
};
}  // namespace cldnn
