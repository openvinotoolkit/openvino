// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "implementation_manager.hpp"
#include "openvino/core/except.hpp"

#include <functional>
#include <memory>
#include <tuple>

namespace cldnn {

template <typename T, typename primitive_type>
class singleton_list : public std::vector<T> {
    singleton_list() : std::vector<T>() {}
    singleton_list(singleton_list const&) = delete;
    void operator=(singleton_list const&) = delete;

public:
    using type = primitive_type;
    static singleton_list& instance() {
        static singleton_list instance_;
        return instance_;
    }
};

template <typename primitive_kind>
class implementation_map {
public:
    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const typed_program_node<primitive_kind>&, const kernel_impl_params&)>;
    using key_type = cldnn::key_type;
    using list_type = singleton_list<std::tuple<impl_types, shape_types, std::shared_ptr<ImplementationManager>>, primitive_kind>;

    static std::shared_ptr<ImplementationManager> get(impl_types preferred_impl_type, shape_types target_shape_type) {
        const auto& l = list_type::instance();
        for (auto& entry : l) {
            impl_types impl_type = std::get<0>(entry);
            if ((preferred_impl_type & impl_type) != impl_type)
                continue;

            shape_types supported_shape_type = std::get<1>(entry);
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;

            return std::get<2>(entry);
        }
        return nullptr;
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
        auto f = std::make_shared<ImplementationManagerLegacy<primitive_kind>>(factory, impl_type, shape_type, keys);
        list_type::instance().push_back({impl_type, shape_type, std::move(f)});
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

}  // namespace cldnn
