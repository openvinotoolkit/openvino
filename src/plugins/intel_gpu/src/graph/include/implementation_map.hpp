// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <typeinfo>


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

struct primitive_impl;

template <class PType>
struct typed_program_node;

struct implementation_key {
    typedef std::tuple<data_types, format::type> type;
    type operator()(const layout& proposed_layout) {
        return std::make_tuple(proposed_layout.data_type, proposed_layout.format);
    }
};

template <typename primitive_kind>
class implementation_map {
public:
    using key_builder = implementation_key;
    using key_type = typename key_builder::type;
    using factory_type = std::function<std::unique_ptr<primitive_impl>(const typed_program_node<primitive_kind>&, const kernel_impl_params&)>;
    using map_type = singleton_map<std::pair<impl_types, shape_types>, std::pair<std::set<key_type>, factory_type>>;

    static factory_type get(const kernel_impl_params& impl_params, impl_types preferred_impl_type, shape_types target_shape_type) {
        auto input_layout = !impl_params.input_layouts.empty() ? impl_params.input_layouts[0] : layout{ov::PartialShape{}, data_types::f32, format::any};
        auto key = key_builder()(input_layout);
        for (auto& kv : map_type::instance()) {
            impl_types impl_type = kv.first.first;
            shape_types supported_shape_type = kv.first.second;
            if ((preferred_impl_type & impl_type) != impl_type)
                continue;
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;
            std::set<key_type>& keys_set = kv.second.first;
            auto& factory = kv.second.second;
            if (keys_set.empty() || keys_set.find(key) != keys_set.end())  {
                return factory;
            }
        }
        OPENVINO_ASSERT(false, "[GPU] implementation_map for ", typeid(primitive_kind).name(),
                               " could not find any implementation to match key: ", std::get<0>(key), "|", std::get<1>(key),
                               ", impl_type: ", preferred_impl_type, ", shape_type: ", target_shape_type, ", node_id: ",  impl_params.desc->id);
    }

    // check if for a given engine and type there exist an implementation
    static bool check(const kernel_impl_params& impl_params, impl_types target_impl_type, shape_types shape_type) {
        auto input_layout = !impl_params.input_layouts.empty() ? impl_params.input_layouts[0] : layout{ov::PartialShape{}, data_types::f32, format::any};
        auto key = key_builder()(input_layout);
        return check_key(target_impl_type, key, shape_type);
    }

    // check if there exists a kernel implementation of a primitive with output set it primitive's output layout
    static bool check_io_eq(const kernel_impl_params& impl_params, impl_types target_impl_type, shape_types shape_type) {
        auto output_layout = !impl_params.output_layouts.empty() ? impl_params.get_output_layout() : layout{ov::PartialShape{}, data_types::f32, format::any};
        auto key = key_builder()(output_layout);
        return check_key(target_impl_type, key, shape_type);
    }

    static bool check_key(impl_types target_impl_type, key_type key, shape_types target_shape_type) {
        for (auto& kv : map_type::instance()) {
            impl_types impl_type = kv.first.first;
            shape_types supported_shape_type = kv.first.second;
            if ((target_impl_type & impl_type) != impl_type)
                continue;
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;
            std::set<key_type>& keys_set = kv.second.first;
            if (keys_set.empty())
                return true;
            return keys_set.find(key) != keys_set.end();
        }
        return false;
    }

    static void add(impl_types impl_type, shape_types shape_type, factory_type factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, shape_type, factory, combine(types, formats));
    }

    static void add(impl_types impl_type, factory_type factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, factory, combine(types, formats));
    }

    static void add(impl_types impl_type, factory_type factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        add(impl_type, shape_types::static_shape, factory, keys);
    }

    static void add(impl_types impl_type, shape_types shape_type, factory_type factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        map_type::instance().insert({{impl_type, shape_type}, {keys, factory}});
    }

private:
    static std::set<key_type> combine(const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        std::set<key_type> keys;
        for (const auto& type : types) {
            for (const auto& format : formats) {
                keys.emplace(type, format);
            }
        }
        return keys;
    }
};
}  // namespace cldnn
