// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief inference engine plugin API wrapper, to be used by particular implementors
 * \file ie_plugin_base.hpp
 */

#pragma once

#include <memory>
#include <map>
#include <string>
#include <blob_factory.hpp>
#include "graph_transformer.h"
#include "ie_util_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "cpp_interfaces/base/ie_executable_network_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "ie_memcpy.h"

#include <details/caseless.hpp>
#include <details/ie_cnn_network_tools.h>
#include <cnn_network_ngraph_impl.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace InferenceEngine {

namespace PluginConfigInternalParams {
    // Quantizer is not used.
    DECLARE_CONFIG_VALUE(QUANTIZE_MODE_NONE);
    // Handle quantize layers only on weights, execute in FP32.
    DECLARE_CONFIG_VALUE(QUANTIZE_MODE_TRANFORM_ON_WEIGHTS_FP32);
    // Handle quantize layers only on weights, execute in INT8.
    DECLARE_CONFIG_VALUE(QUANTIZE_MODE_TRANFORM_ON_WEIGHTS_INT8);
    // Handle quantize layers on weights and activations, execute in FP32.
    DECLARE_CONFIG_VALUE(QUANTIZE_MODE_TRANFORM_ON_WEIGHTS_AND_DATA_FP32);
    // Handle quantize layers on weights and activations, execute in INT8. Default value.
    DECLARE_CONFIG_VALUE(QUANTIZE_MODE_TRANFORM_ON_WEIGHTS_AND_DATA_INT8);

    DECLARE_CONFIG_KEY(QUANTIZE_MODE);
}  // namespace PluginConfigInternalParams

/**
 * @brief optional implementation of IInferencePluginInternal to avoid duplication in all plugins
 */
class InferencePluginInternal
        : public IInferencePluginInternal, public std::enable_shared_from_this<InferencePluginInternal> {
public:
    /**
     * @brief most plugins successfully consume unreshapable networks - lets do it in base class
     * WARNING: this functions modifies layers in input network and might affect application, that uses it
     */
    virtual ICNNNetwork&  RemoveConstLayers(ICNNNetwork &network) {
        auto* implNetwork = dynamic_cast<details::CNNNetworkImpl*>(&network);
        if (implNetwork) {
            // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
            ConstTransformer transformator(implNetwork);
            transformator.fullTrim();
        }
        return network;
    }

    /**
     * @brief clone network
     */
    virtual InferenceEngine::details::CNNNetworkImplPtr CloneNetwork(const InferenceEngine::ICNNNetwork& network) {
        return cloneNet(network);
    }

    /**
     * @brief move blobs from Constant layers to Convolution and FullyConnected layers attributes
     * WARNING: this functions modifies layers in input network and might affect application, that uses it
     */
    virtual ICNNNetwork&  MoveWeights(ICNNNetwork &network) {
        details::CNNNetworkImpl* networkImpl = dynamic_cast<details::CNNNetworkImpl*>(&network);
        if (networkImpl != nullptr) {
            ConstTransformer constTransformer(networkImpl);
            constTransformer.moveWeights();
        }
        return network;
    }

    /**
     * @brief Creates an executable network from an pares network object, users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the HW resources)
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     * @return shared_ptr to the ExecutableNetwork object
     */
    virtual ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const ICore * core, ICNNNetwork &network,
                                                              const std::map<std::string, std::string> &config) = 0;

    /**
     * Given optional implementation of load executable network to avoid need for it to be implemented by plugin
     */
    void LoadNetwork(IExecutableNetwork::Ptr &executableNetwork,
                     ICNNNetwork& network,
                     const std::map<std::string, std::string> &config) override {
        CNNNetworkImplPtr clonedNetworkImpl = std::dynamic_pointer_cast<CNNNetworkImpl>(ConvertAndCloneNetwork(network));

        InputsDataMap networkInputs;
        OutputsDataMap networkOutputs;
        clonedNetworkImpl->getInputsInfo(networkInputs);
        clonedNetworkImpl->getOutputsInfo(networkOutputs);
        _networkInputs.clear();
        _networkOutputs.clear();

        for (const auto& it : networkInputs) {
            InputInfo::Ptr newPtr;
            if (it.second) {
                newPtr.reset(new InputInfo());
                DataPtr newData(new Data(*it.second->getInputData()));
                newPtr->getPreProcess() = it.second->getPreProcess();
                if (newPtr->getPreProcess().getMeanVariant() == MEAN_IMAGE) {
                    for (size_t i = 0; i < newPtr->getPreProcess().getNumberOfChannels(); i++) {
                        auto blob = newPtr->getPreProcess()[i]->meanData;
                        newPtr->getPreProcess()[i]->meanData =
                                make_blob_with_precision(newPtr->getPreProcess()[i]->meanData->getTensorDesc());
                        newPtr->getPreProcess()[i]->meanData->allocate();
                        ie_memcpy(newPtr->getPreProcess()[i]->meanData->buffer(), newPtr->getPreProcess()[i]->meanData->byteSize(),
                                  blob->cbuffer(), blob->byteSize());
                    }
                }
                newData->getInputTo().clear();
                newPtr->setInputData(newData);
            }
            _networkInputs[it.first] = newPtr;
        }

        for (const auto& it : networkOutputs) {
            DataPtr newData;
            if (it.second) {
                newData.reset(new Data(*it.second));
                newData->getInputTo().clear();
            }
            _networkOutputs[it.first] = newData;
        }

        // move blobs from Constant layers to Convolution and FullyConnected layers attributes
        MoveWeights(*clonedNetworkImpl);

        auto impl = LoadExeNetworkImpl(GetCore(), RemoveConstLayers(*clonedNetworkImpl), config);
        impl->setNetworkInputs(_networkInputs);
        impl->setNetworkOutputs(_networkOutputs);
        impl->SetPointerToPluginInternal(shared_from_this());

        executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(impl), [](details::IRelease *p) {
            p->Release();
        });
    }

    /**
     * Given optional implementation of ImportNetwork to avoid need for it to be implemented by plugin
     */
    IExecutableNetwork::Ptr ImportNetwork(const std::string &/*modelFileName*/, const std::map<std::string, std::string> &/*config*/) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    /**
     * Given optional implementation of SetConfig to avoid need for it to be implemented by plugin
     */
    void SetConfig(const std::map<std::string, std::string> &config) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    /**
     * Given optional implementation of SetLogCallback to avoid need for it to be implemented by plugin
     */
    void SetLogCallback(IErrorListener &/*listener*/) override {}

    /**
     * Given optional implementation of SetLogCallback to avoid need for it to be implemented by plugin
     */
    void SetCore(ICore* core) noexcept override {
        assert(nullptr != core);
        _core = core;
    }

    /**
     * Given optional implementation of SetLogCallback to avoid need for it to be implemented by plugin
     */
    const ICore* GetCore() const noexcept override {
        return _core;
    }

    /**
     * Given optional implementation of AddExtension to avoid need for it to be implemented by plugin
     */
    void AddExtension(InferenceEngine::IExtensionPtr /*extension*/) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void QueryNetwork(const ICNNNetwork &/*network*/, const std::map<std::string, std::string>& /*config*/, QueryNetworkResult &/*res*/) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void SetName(const std::string & pluginName) noexcept override {
        _pluginName = pluginName;
    }

    std::string GetName() const noexcept override {
        return _pluginName;
    }

    Parameter GetConfig(const std::string& /*name*/, const std::map<std::string, Parameter> & /*options*/) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    Parameter GetMetric(const std::string& /*name*/, const std::map<std::string, Parameter> & /*options*/) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

protected:
    // Input network preprocessing before regular network loading
    virtual std::shared_ptr<ICNNNetwork> ConvertAndCloneNetwork(ICNNNetwork& network) {
#ifdef ENABLE_NGRAPH
        if (auto networkNGraph = dynamic_cast<CNNNetworkNGraphImpl*>(&network)) {
            return networkNGraph->convertToCNNNetworkImpl();
        }
#endif
        return CloneNetwork(network);
    }

    std::string _pluginName;
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    std::map<std::string, std::string> _config;
    ICore*   _core = nullptr;
};

}  // namespace InferenceEngine
