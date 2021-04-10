// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iexecutable_network.hpp>
#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <ie_iinfer_request.hpp>
#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {
class IInferRequestInternal;
/**
 * @interface IExecutableNetworkInternal
 * @brief An internal API of executable network to be implemented by plugin,
 * @ingroup ie_dev_api_exec_network_api
 */
class IExecutableNetworkInternal : public std::enable_shared_from_this<IExecutableNetworkInternal> {
public:
    /**
     * @brief A shared pointer to IExecutableNetworkInternal interface
     */
    using Ptr = std::shared_ptr<IExecutableNetworkInternal>;

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

    /**
     * @brief Gets the Executable network output Data node information. The received info is stored in the given Data
     * node. This method need to be called to find output names for using them later during filling of a map of blobs
     * passed later to InferenceEngine::IInferencePlugin::Infer()
     * @return out Reference to the ConstOutputsDataMap object
     */
    virtual ConstOutputsDataMap GetOutputsInfo() const {
        ConstOutputsDataMap outputMap;
        for (const auto& output : _networkOutputs) {
            outputMap[output.first] = output.second;
        }
        return outputMap;
    }

    /**
     * @brief Gets the Executable network input Data node information. The received info is stored in the given
     * InputsDataMap object. This method need to be called to find out input names for using them later during filling
     * of a map of blobs passed later to InferenceEngine::IInferencePlugin::Infer()
     * @return inputs Reference to ConstInputsDataMap object.
     */
    virtual ConstInputsDataMap GetInputsInfo() const {
        ConstInputsDataMap inputMap;
        for (const auto& input : _networkInputs) {
            inputMap.emplace(input.first, input.second);
        }
        return inputMap;
    }

    /**
     * @brief Create an inference request object used to infer the network
     *  Note: the returned request will have allocated input and output blobs (that can be changed later)
     * @return shared_ptr for the created request
     */
    virtual std::shared_ptr<IInferRequestInternal> CreateInferRequest() {
        IInferRequest::Ptr asyncRequest;
        auto asyncRequestImpl = this->CreateInferRequestImpl(_networkInputs, _networkOutputs);
        asyncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        return asyncRequestImpl;
    }

    /**
     * @deprecated Use IExecutableNetworkInternal::Export(std::ostream& networkModel)
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param modelFileName - path to the location of the exported file
     */
    virtual void Export(const std::string& modelFileName) {
        // we need to write to stringstream first
        // because in case of exception in ExportImpl the file is not created
        std::stringstream strm;
        ExportImpl(strm);
        std::ofstream(modelFileName.c_str()) << strm.rdbuf();
    }

    /**
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param networkModel - Reference to network model output stream
     */
    virtual void Export(std::ostream& networkModel) {
        std::stringstream strm;
        strm.write(exportMagic.data(), exportMagic.size());
        strm << _plugin->GetName() << std::endl;
        ExportImpl(strm);
        networkModel << strm.rdbuf();
    }

    /**
     * @brief Get executable graph information from a device
     * @return A network object to store executable graph information
     */
    virtual CNNNetwork GetExecGraphInfo()  {
        IE_THROW(NotImplemented);
    }

    /**
     * @brief      Sets the pointer to plugin internal.
     * @param[in]  plugin  The plugin
     * @note Needed to correctly handle ownership between objects.
     */
    void SetPointerToPlugin(IInferencePlugin::Ptr plugin) {
        _plugin = plugin;
    }

    /**
     * @deprecated Need to implement GetVariablesInfo for ExecutableNetwork
     * @brief Queries memory states.
     * @return Returns memory states
     */
    virtual std::vector<IVariableStateInternal::Ptr> QueryState() {
        IE_THROW(NotImplemented);
    }

    /**
     * @brief Sets configuration for current executable network
     * @param config Map of pairs: (config parameter name, config parameter value)
     */
    virtual void SetConfig(const std::map<std::string, Parameter>& config) {
        if (config.empty()) {
            IE_THROW() << "The list of configuration values is empty";
        }
        IE_THROW() << "The following config value cannot be changed dynamically for ExecutableNetwork: "
                           << config.begin()->first;
    }

    /**
     * @brief Gets configuration dedicated to plugin behaviour
     * @param name A config key, can be found in ie_plugin_config.hpp
     * @return A value of config corresponding to config key
     */
    virtual Parameter GetConfig(const std::string& name) const {
        (void)name;
        IE_THROW() << "GetConfig for executable network is not supported by this device";
    }

    /**
     * @brief Gets general runtime metric for dedicated hardware
     * @param name  A metric name to request
     * @return A metric value corresponding to metric key
     */
    virtual Parameter GetMetric(const std::string& name) const {
        (void)name;
        IE_THROW(NotImplemented);
    }

    /**
     * @brief Gets the remote context.
     * @return A reference to a context
     */
    virtual RemoteContext::Ptr GetContext() const {
        IE_THROW(NotImplemented);
    }

protected:
    /**
     * @brief      Destroys the object.
     */
    ~IExecutableNetworkInternal() = default;

    /**
     * @brief      Creates an asynchronous inference request internal implementation.
     * @note       The method is called by ExecutableNetworkInternal::CreateInferRequest as
     *             plugin-specific implementation.
     * @param[in]  networkInputs   The network inputs
     * @param[in]  networkOutputs  The network outputs
     * @return     A shared pointer to asynchnous inference request object.
     */
    virtual IInferRequestInternal::Ptr CreateInferRequestImpl(InputsDataMap networkInputs,
                                                              OutputsDataMap networkOutputs) {
        IE_THROW(NotImplemented);
    }

    /**
     * @brief Exports an internal hardware-dependent model to a stream.
     * @note The function is called from ExecutableNetworkInternal::Export(std::ostream&),
     * which performs common export first and calls this plugin-dependent implementation after.
     * @param networkModel A stream to export network to.
     */
    virtual void ExportImpl(std::ostream& networkModel) {
        (void)networkModel;
        IE_THROW(NotImplemented);
    }

    InferenceEngine::InputsDataMap _networkInputs;  //!< Holds information about network inputs info
    InferenceEngine::OutputsDataMap _networkOutputs;  //!< Holds information about network outputs data

    std::unordered_map<std::string, ITaskExecutor::Ptr> _executors;

    /**
     * @brief A pointer to a IInferencePlugin interface.
     * @note Needed to correctly handle ownership between objects.
     */
    IInferencePlugin::Ptr _plugin;
};

}  // namespace InferenceEngine
