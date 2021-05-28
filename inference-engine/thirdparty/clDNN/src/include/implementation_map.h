// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <functional>
#include <typeinfo>
#include <tuple>
#include <string>

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

namespace cldnn {

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
    typedef std::tuple<engine_types, data_types, format::type> type;
    type operator()(engine_types engine_type, const typed_program_node<primitive_kind>& primitive) {
        return std::make_tuple(engine_type,
                               primitive.get_dependency(0).get_output_layout().data_type,
                               primitive.get_dependency(0).get_output_layout().format);
    }
    type operator()(engine_types engine_type, const layout& proposed_layout) {
        return std::make_tuple(engine_type, proposed_layout.data_type, proposed_layout.format);
    }
};

template <>
struct implementation_key<permute> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<permute>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<reorder> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<reorder>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<generic_layer> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<generic_layer>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<custom_gpu_primitive> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<custom_gpu_primitive>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<reshape> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<reshape>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<data> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<data>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<mutable_data> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<mutable_data>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<input_layout> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<input_layout>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<prior_box> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<prior_box>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <>
struct implementation_key<loop> {
    typedef cldnn::engine_types type;
    type operator()(engine_types engine_type, const typed_program_node<loop>&) { return engine_type; }
    type operator()(engine_types engine_type, const layout&) { return engine_type; }
};

template <typename primitive_kind>
class implementation_map {
public:
    using key_builder = implementation_key<primitive_kind>;
    using key_type = typename key_builder::type;
    using factory_type = std::function<primitive_impl*(const typed_program_node<primitive_kind>&)>;
    using map_type = singleton_map<key_type, factory_type>;

    static factory_type get(engine_types engine_type, const typed_program_node<primitive_kind>& primitive) {
        // lookup in database; throw if not found
        auto key = key_builder()(engine_type, primitive);
        auto it = map_type::instance().find(key);
        if (it == std::end(map_type::instance()))
            throw std::runtime_error(std::string("implementation_map for ") + typeid(primitive_kind).name() +
                                     " could not find any implementation to match key");
        // create implementation & attach it to result
        return it->second;
    }

    // check if for a given engine and type there exist an implementation
    static bool check(engine_types engine_type, const typed_program_node<primitive_kind>& primitive) {
        auto key = key_builder()(engine_type, primitive);
        auto it = map_type::instance().find(key);
        if (it == std::end(map_type::instance()))
            return false;
        else
            return true;
    }

    // check if there exists a kernel implementation of a primitive with output set it primitive's output layout
    static bool check_io_eq(engine_types engine_type, const typed_program_node<primitive_kind>& primitive) {
        auto key = key_builder()(engine_type, primitive.get_output_layout());
        auto it = map_type::instance().find(key);
        if (it == std::end(map_type::instance()))
            return false;
        else
            return true;
    }

    static void add(typename map_type::key_type key, factory_type factory) {
        map_type::instance().insert({key, factory});
    }

    static void add(std::initializer_list<typename map_type::value_type> il) { map_type::instance().insert(il); }
};
}  // namespace cldnn
