// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace InferenceEngine {

/**
 * @brief Class wraps InferenceEngine::IInferencePlugin into ov::IPlugin
 */
class IPluginWrapper : public ov::IPlugin {
public:
    /**
     * @brief Constructs Plugin wrapper
     *
     * @param ptr shared pointer to InferenceEngine::IInferencePlugin
     */
    IPluginWrapper(const std::shared_ptr<InferenceEngine::IInferencePlugin>& ptr);

    /**
     * @brief Destructor
     */
    virtual ~IPluginWrapper() = default;

    /**
     * @brief Create compiled model based on model and properties
     *
     * @param model OpenVINO Model representation
     * @param properties configurations for compiled model
     *
     * @return shared pointer to compiled model interface
     */
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    /**
     * @brief Create compiled model based on model and properties
     *
     * @param model_path Path to the model
     * @param properties configurations for compiled model
     *
     * @return shared pointer to compiled model interface
     */
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::string& model_path,
                                                      const ov::AnyMap& properties) const override;

    /**
     * @brief Create compiled model based on model and properties
     *
     * @param model OpenVINO Model representation
     * @param properties configurations for compiled model
     * @param context remote context
     *
     * @return shared pointer to compiled model interface
     */
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    /**
     * @brief Specifies some plugin properties
     *
     * @param properties map with configuration properties
     */
    void set_property(const ov::AnyMap& properties) override;

    /**
     * @brief Returns the property
     *
     * @param name property name
     * @param arguments configuration parameters
     *
     * @return ov::Any object which contains property value
     */
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    /**
     * @brief Create remote context
     *
     * @param remote_properties configuration parameters
     *
     * @return Remote context
     */
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    /**
     * @brief Create default remote context
     *
     * @param remote_properties configuration parameters
     *
     * @return Remote context
     */
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    /**
     * @brief Import model to the plugin
     *
     * @param model strim with the model
     * @param properties configuration properties
     *
     * @return shared pointer to compiled model interface
     */
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    /**
     * @brief Import model to the plugin
     *
     * @param model strim with the model
     * @param context remote context
     * @param properties configuration properties
     *
     * @return shared pointer to compiled model interface
     */
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    /**
     * @brief query model
     *
     * @param model OpenVINO Model
     * @param properties configuration properties
     *
     * @return Map of supported operations
     */
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    /**
     * @brief Register legacy Inference Engine Extension for the plugin
     *
     * @param extension legacy Inference Engine Extension
     */
    void add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension) override;

    /**
     * @brief Returns the instance of the legacy plugin
     *
     * @return Legacy InferenceEngine::IInferencePlugin object
     */
    const std::shared_ptr<InferenceEngine::IInferencePlugin>& get_plugin() const;

    /**
     * @brief Set core interface to the plugin
     * This method works under the non-virtual method of IPlugin class
     *
     * @param core OpenVINO Core interface
     */
    void set_core(const std::weak_ptr<ov::ICore>& core);

    /**
     * @brief Set plugin name for the wrapper and legacy plugin
     * This method works under the non-virtual method of IPlugin class
     *
     * @param device_name The name of plugin
     */
    void set_device_name(const std::string& device_name);

    void set_shared_object(const std::shared_ptr<void>& so);

private:
    std::shared_ptr<InferenceEngine::IInferencePlugin> m_old_plugin;
    std::shared_ptr<void> m_so;

    const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& update_exec_network(
        const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& network) const;
};

}  // namespace InferenceEngine
