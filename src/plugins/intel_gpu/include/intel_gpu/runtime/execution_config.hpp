// Copyright (C) 2018-2022 Intel Corporation
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
    /**
     * @brief Sets properties for the current compiled model.
     *
     * @param properties Map of pairs: (property name, property value).
     */
    void set_property(const AnyMap& properties);

    /**
     * @brief Sets properties for the current compiled model.
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types.
     * @param properties Optional pack of pairs: (property name, property value).
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(AnyMap{std::forward<Properties>(properties)...});
    }

    /** @brief Gets properties for current compiled model
     *
     * The method is responsible for extracting information
     * that affects compiled model inference. The list of supported configuration values can be extracted via
     * CompiledModel::get_property with the ov::supported_properties key, but some of these keys cannot be changed
     * dynamically, for example, ov::device::id cannot be changed if a compiled model has already been compiled for a
     * particular device.
     *
     * @param name Property key, can be found in openvino/runtime/properties.hpp.
     * @return Property value.
     */
    Any get_property(const std::string& name) const;

    /**
     * @brief Gets properties related to device behaviour.
     *
     * The method extracts information that can be set via the set_property method.
     *
     * @tparam T Type of a returned value.
     * @param property  Property  object.
     * @return Value of property.
     */
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
