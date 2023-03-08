// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine plugin C++ API
 *
 * @file plugin.hpp
 */
#pragma once

#include "ie_iextension.h"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace ov {

class CoreImpl;

/**
 * @brief Plugin wrapper under the plugin interface which is used inside the core interface
 */
class Plugin {
    std::shared_ptr<ov::IPlugin> m_ptr;
    std::shared_ptr<void> m_so;
    friend ::ov::CoreImpl;

public:
    Plugin() = default;

    ~Plugin();

    Plugin(const std::shared_ptr<ov::IPlugin>& ptr, const std::shared_ptr<void>& so);

    void set_name(const std::string& deviceName);

    const std::string& get_name() const;

    void set_core(std::weak_ptr<ICore> core);

    const ov::Version get_version() const;

    void add_extension(const ie::IExtensionPtr& extension);

    void set_property(const ov::AnyMap& config);

    SoPtr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::AnyMap& properties) const;

    SoPtr<ov::ICompiledModel> compile_model(const std::string& model_path, const ov::AnyMap& properties) const;

    SoPtr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                            const ov::RemoteContext& context,
                                            const ov::AnyMap& properties) const;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& properties) const;

    SoPtr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const;

    SoPtr<ov::ICompiledModel> import_model(std::istream& networkModel,
                                           const ov::RemoteContext& context,
                                           const ov::AnyMap& config) const;

    ov::RemoteContext create_context(const AnyMap& params) const;

    ov::RemoteContext get_default_context(const AnyMap& params) const;

    Any get_property(const std::string& name, const AnyMap& arguments) const;

    template <typename T, PropertyMutability M>
    T get_property(const ov::Property<T, M>& property) const {
        return get_property(property.name(), {}).template as<T>();
    }

    template <typename T, PropertyMutability M>
    T get_property(const ov::Property<T, M>& property, const AnyMap& arguments) const {
        return get_property(property.name(), arguments).template as<T>();
    }
};

}  // namespace ov

