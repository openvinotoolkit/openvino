// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include "cpp_interfaces/impl/ie_infer_async_request_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

namespace InferenceEngine {

/**
 * @brief Minimum implementation of IExecutableNetworkInternal interface. Must not be used as a base class in plugins
 * As base classes, use ExecutableNetworkThreadSafeDefault or ExecutableNetworkThreadSafeAsyncOnly
 * @ingroup ie_dev_api_exec_network_api
 */
class ExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    /**
     * @brief A shared pointer to ExecutableNetworkInternal object
     */
    typedef std::shared_ptr<ExecutableNetworkInternal> Ptr;

    /**
     * @brief      Sets the network inputs info.
     * @param[in]  networkInputs  The network inputs info
     */
    virtual void setNetworkInputs(const InferenceEngine::InputsDataMap networkInputs) {
        _networkInputs = networkInputs;
    }

    /**
     * @brief      Sets the network outputs data.
     * @param[in]  networkOutputs  The network outputs
     */
    virtual void setNetworkOutputs(const InferenceEngine::OutputsDataMap networkOutputs) {
        _networkOutputs = networkOutputs;
    }

    ConstOutputsDataMap GetOutputsInfo() const override {
        ConstOutputsDataMap outputMap;
        for (const auto& output : _networkOutputs) {
            outputMap[output.first] = output.second;
        }
        return outputMap;
    }

    ConstInputsDataMap GetInputsInfo() const override {
        ConstInputsDataMap inputMap;
        for (const auto& input : _networkInputs) {
            inputMap[input.first] = input.second;
        }
        return inputMap;
    }

    void Export(const std::string& modelFileName) override {
        // we need to write to stringstream first
        // because in case of exception in ExportImpl the file is not created
        std::stringstream strm;
        ExportImpl(strm);
        std::ofstream(modelFileName.c_str()) << strm.rdbuf();
    }

    void Export(std::ostream& networkModel) override {
        std::stringstream strm;
        strm.write(exportMagic.data(), exportMagic.size());
        strm << _plugin->GetName() << std::endl;
        ExportImpl(strm);
        networkModel << strm.rdbuf();
    }

    CNNNetwork GetExecGraphInfo() override {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }

    /**
     * @brief      Sets the pointer to plugin internal.
     * @param[in]  plugin  The plugin
     * @note Needed to correctly handle ownership between objects.
     */
    void SetPointerToPlugin(IInferencePlugin::Ptr plugin) {
        _plugin = plugin;
    }

    std::vector<IVariableStateInternal::Ptr> QueryState() override {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }

    void SetConfig(const std::map<std::string, Parameter>& config) override {
        if (config.empty()) {
            THROW_IE_EXCEPTION << "The list of configuration values is empty";
        }
        THROW_IE_EXCEPTION << "The following config value cannot be changed dynamically for ExecutableNetwork: "
                           << config.begin()->first;
    }

    Parameter GetConfig(const std::string& name) const override {
        (void)name;
        THROW_IE_EXCEPTION << "GetConfig for executable network is not supported by this device";
    }

    Parameter GetMetric(const std::string& name) const override {
        (void)name;
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }

    RemoteContext::Ptr GetContext() const override {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }

protected:
    /**
     * @brief Exports an internal hardware-dependent model to a stream.
     * @note The function is called from ExecutableNetworkInternal::Export(std::ostream&),
     * which performs common export first and calls this plugin-dependent implementation after.
     * @param networkModel A stream to export network to.
     */
    virtual void ExportImpl(std::ostream& networkModel) {
        (void)networkModel;
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED);
    }

    InferenceEngine::InputsDataMap _networkInputs;  //!< Holds infromation about network inputs info
    InferenceEngine::OutputsDataMap _networkOutputs;  //!< Holds information about network outputs data

    /**
     * @brief A pointer to a IInferencePlugin interface.
     * @note Needed to correctly handle ownership between objects.
     */
    IInferencePlugin::Ptr _plugin;
};

}  // namespace InferenceEngine
