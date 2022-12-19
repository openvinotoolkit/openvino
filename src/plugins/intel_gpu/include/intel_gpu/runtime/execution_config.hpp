// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/internal_properties.hpp"

namespace ov {
namespace intel_gpu {

class ExecutionConfig {
public:
    ExecutionConfig();
    ExecutionConfig(std::initializer_list<ov::AnyMap::value_type> values) : ExecutionConfig() { set_property(ov::AnyMap(values)); }
    explicit ExecutionConfig(const ov::AnyMap& properties) : ExecutionConfig() { set_property(properties); }
    explicit ExecutionConfig(const ov::AnyMap::value_type& property) : ExecutionConfig() { set_property(property); }

    void set_property(const AnyMap& properties);

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(AnyMap{std::forward<Properties>(properties)...});
    }

    Any get_property(const std::string& name) const;

    template <typename T, PropertyMutability mutability>
    T get_property(const ov::Property<T, mutability>& property) const {
        return get_property(property.name()).template as<T>();
    }

    std::string to_string() const {
        std::stringstream s;
        s << "Config\n";
        for (auto& kv : properties) {
            s << "\t" << kv.first << ": " << kv.second.as<std::string>() << std::endl;
        }

        return s.str();
    }

private:
    AnyMap properties;
};

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::ExecutionConfig;
}  // namespace cldnn
