// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"

#include <functional>
#include <string>
#include <tuple>
#include <typeinfo>


namespace cldnn {

template <typename T>
class singleton_list : public std::vector<T> {
    singleton_list() : std::vector<T>() {}
    singleton_list(singleton_list const&) = delete;
    void operator=(singleton_list const&) = delete;

public:
    static singleton_list& instance() {
        static singleton_list instance_;
        return instance_;
    }
};

using in_out_fmts_t = std::pair<std::vector<format::type>, std::vector<format::type>>;

struct primitive_impl;

struct program_node;
template <class PType>
struct typed_program_node;

struct implementation_key {
    typedef std::tuple<data_types, format::type> type;
    type operator()(const layout& proposed_layout) {
        return std::make_tuple(proposed_layout.data_type, proposed_layout.format);
    }
};

struct implementation_factory_base {
public:
    virtual std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual bool validate(const program_node& node) const = 0;
    virtual in_out_fmts_t query_formats(const program_node& node) const = 0;

    virtual ~implementation_factory_base() = default;
};

template <typename primitive_kind>
struct implementation_factory : public implementation_factory_base {
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const override {
        if (f)
            return f(static_cast<const typed_program_node<primitive_kind>&>(node), params);

        OPENVINO_NOT_IMPLEMENTED;
    };
    bool validate(const program_node& node) const override {
        return true;
    }
    in_out_fmts_t query_formats(const program_node& node) const override {
        return {};
    }
    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const typed_program_node<primitive_kind>&, const kernel_impl_params&)>;
    explicit implementation_factory(simple_factory_type factory) : f(factory) { }
    implementation_factory() = default;

private:
    simple_factory_type f;
};

template <typename primitive_kind>
class implementation_map {
public:
    using key_builder = implementation_key;
    using key_type = typename key_builder::type;
    using factory_type = implementation_factory<primitive_kind>;
    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const typed_program_node<primitive_kind>&, const kernel_impl_params&)>;
    using validator_type = std::function<bool(const typed_program_node<primitive_kind>&)>;
    using list_type = singleton_list<std::tuple<impl_types, shape_types, std::set<key_type>, std::unique_ptr<factory_type>>>;

    static const factory_type* get(const kernel_impl_params& impl_params, impl_types preferred_impl_type, shape_types target_shape_type) {
        auto input_layout = !impl_params.input_layouts.empty() ? impl_params.input_layouts[0] : layout{ov::PartialShape{}, data_types::f32, format::any};
        auto key = key_builder()(input_layout);
        for (auto& kv : list_type::instance()) {
            impl_types impl_type = std::get<0>(kv);
            shape_types supported_shape_type = std::get<1>(kv);
            if ((preferred_impl_type & impl_type) != impl_type)
                continue;
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;
            std::set<key_type>& keys_set = std::get<2>(kv);
            auto& factory = std::get<3>(kv);
            if (keys_set.empty() || keys_set.find(key) != keys_set.end())  {
                return factory.get();
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
        for (auto& kv : list_type::instance()) {
            impl_types impl_type = std::get<0>(kv);
            shape_types supported_shape_type = std::get<1>(kv);
            if ((target_impl_type & impl_type) != impl_type)
                continue;
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;
            std::set<key_type>& keys_set = std::get<2>(kv);
            if (keys_set.empty())
                return true;
            return keys_set.find(key) != keys_set.end();
        }
        return false;
    }

    static bool is_impl_supported(const typed_program_node<primitive_kind>& node, impl_types impl_type) {
        const auto& impls = list_type::instance();
        auto desc = std::find_if(impls.begin(), impls.end(), [&impl_type](const typename list_type::value_type& v) {
            return std::get<0>(v) == impl_type;
        });
        if (desc == impls.end())
            return false;

        return std::get<3>(*desc)->validate(node);
    }

    static std::set<impl_types> query_available_impls(data_types in_dt, shape_types target_shape_type, const typed_program_node<primitive_kind>& node) {
        std::set<impl_types> res;
        for (auto& kv : list_type::instance()) {
            impl_types impl_type = std::get<0>(kv);
            const auto& factory = std::get<3>(kv);
            shape_types supported_shape_type = std::get<1>(kv);
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;
            if (!factory->validate(node))
                continue;

            std::set<key_type>& keys_set = std::get<2>(kv);
            for (const auto& key : keys_set) {
                if (std::get<0>(key) == in_dt) {
                    res.insert(impl_type);
                    break;
                }
            }
            if (keys_set.empty()) {
                res.insert(impl_type);
            }
        }
        return res;
    }

    static void add(impl_types impl_type, shape_types shape_type, simple_factory_type factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, shape_type, std::move(factory), combine(types, formats));
    }

    static void add(impl_types impl_type, simple_factory_type factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, std::move(factory), combine(types, formats));
    }

    static void add(impl_types impl_type, simple_factory_type factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        add(impl_type, shape_types::static_shape, std::move(factory), keys);
    }

    static void add(impl_types impl_type, shape_types shape_type, simple_factory_type factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        auto f = cldnn::make_unique<implementation_factory<primitive_kind>>(factory);
        list_type::instance().push_back({impl_type, shape_type, keys, std::move(f)});
    }

    static void add(impl_types impl_type, shape_types shape_type, std::unique_ptr<factory_type> factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, shape_type, std::move(factory), combine(types, formats));
    }

    static void add(impl_types impl_type, std::unique_ptr<factory_type> factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, std::move(factory), combine(types, formats));
    }

    static void add(impl_types impl_type, std::unique_ptr<factory_type> factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        add(impl_type, shape_types::static_shape, std::move(factory), keys);
    }

    static void add(impl_types impl_type, shape_types shape_type, std::unique_ptr<factory_type> factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        list_type::instance().push_back({impl_type, shape_type, keys, std::move(factory)});
    }

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

struct WeightsReordersFactory {
    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const kernel_impl_params&)>;
    using list_type = singleton_list<std::tuple<impl_types, shape_types, simple_factory_type>>;
    static void add(impl_types impl_type, shape_types shape_type, simple_factory_type factory) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register WeightsReordersFactory with type any");
        list_type::instance().push_back({impl_type, shape_type, factory});
    }

    static simple_factory_type get(impl_types preferred_impl_type, shape_types target_shape_type) {
        for (auto& kv : list_type::instance()) {
            impl_types impl_type = std::get<0>(kv);
            shape_types supported_shape_type = std::get<1>(kv);
            if ((preferred_impl_type & impl_type) != impl_type)
                continue;
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;

            return std::get<2>(kv);
        }
        OPENVINO_THROW("[GPU] WeightsReordersFactory doesn't have any implementation for "
                       " impl_type: ", preferred_impl_type, ", shape_type: ", target_shape_type);
    }
};
}  // namespace cldnn
