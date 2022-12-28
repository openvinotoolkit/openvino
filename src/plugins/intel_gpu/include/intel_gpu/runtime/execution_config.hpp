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

    void set_default();
    void set_property(const AnyMap& properties);
    void set_user_property(const AnyMap& properties);
    Any get_property(const std::string& name) const;
    bool has_property(std::string name) const;
    bool is_set_by_user(std::string name) const;

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_user_property(Properties&&... properties) {
        set_user_property(AnyMap{std::forward<Properties>(properties)...});
    }

    template <typename T, PropertyMutability mutability>
    bool is_set_by_user(const ov::Property<T, mutability>& property) const {
        return is_set_by_user(property.name());
    }

    template <typename T, PropertyMutability mutability>
    bool has_property(const ov::Property<T, mutability>& property) const {
        return has_property(property.name());
    }

    template <typename T, PropertyMutability mutability>
    T get_property(const ov::Property<T, mutability>& property) const {
        return get_property(property.name()).template as<T>();
    }

    void apply_user_properties();
    void apply_hints();
    void apply_performance_hints();
    void apply_priority_hints();

    std::string to_string() const;

private:
    AnyMap internal_properties;
    AnyMap user_properties;
};

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::ExecutionConfig;
}  // namespace cldnn
