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
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "cpp_interfaces/base/ie_executable_network_base.hpp"
#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "ie_memcpy.h"

namespace InferenceEngine {

/**
 * @brief optional implementation of IInferencePluginInternal to avoid duplication in all plugins
 */
class InferencePluginInternal
        : public IInferencePluginInternal, public std::enable_shared_from_this<InferencePluginInternal> {
public:
    /**
     * Given optional implementation of deprecated load to avoid need for it to be implemented by plugin
     */
    void LoadNetwork(ICNNNetwork &network) override {
        _isDeprecatedLoad = true;
        network.getInputsInfo(_networkInputs);
        network.getOutputsInfo(_networkOutputs);
        if (_networkInputs.empty() || _networkOutputs.empty()) {
            THROW_IE_EXCEPTION << "The network doesn't have inputs/outputs.";
        }
        _createdInferRequest = nullptr;  // first release the infer request
        _loadedNetwork = nullptr;  // first release the loaded network

        _firstInput = _networkInputs.begin()->first;
        _firstOutput = _networkOutputs.begin()->first;
        LoadNetwork(_loadedNetwork, network, {});

        ResponseDesc resp;
        StatusCode sts = _loadedNetwork->CreateInferRequest(_createdInferRequest, &resp);
        if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
    }
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
     * @brief Creates an executable network from an pares network object, users can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the HW resources)
     * @param network - a network object acquired from CNNNetReader
     * @param config string-string map of config parameters relevant only for this load operation
     * @return shared_ptr to the ExecutableNetwork object
     */
    virtual ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(ICNNNetwork &network,
                                                              const std::map<std::string, std::string> &config) = 0;

    /**
     * Given optional implementation of load executable network to avoid need for it to be implemented by plugin
     */
    void LoadNetwork(IExecutableNetwork::Ptr &executableNetwork,
                     ICNNNetwork &network,
                     const std::map<std::string, std::string> &config) override {
        InputsDataMap networkInputs;
        OutputsDataMap networkOutputs;
        network.getInputsInfo(networkInputs);
        network.getOutputsInfo(networkOutputs);
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
                newData->inputTo.clear();
                newPtr->setInputData(newData);
            }
            _networkInputs[it.first] = newPtr;
        }

        for (const auto& it : networkOutputs) {
            DataPtr newData;
            if (it.second) {
                newData.reset(new Data(*it.second));
                newData->inputTo.clear();
            }
            _networkOutputs[it.first] = newData;
        }
        auto impl = LoadExeNetworkImpl(RemoveConstLayers(network), config);
        impl->setNetworkInputs(_networkInputs);
        impl->setNetworkOutputs(_networkOutputs);
        // skip setting shared ptr to avoid curricular dependency: ExecutableNetworkBase -> IExecutableNetworkInternal -> InferencePluginInternal
        if (!_isDeprecatedLoad) {
            impl->SetPointerToPluginInternal(shared_from_this());
        }

        executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(impl), [](details::IRelease *p) {
            p->Release();
        });
        _isDeprecatedLoad = false;
    };

    /**
     * Given optional implementation of deprecated infer to avoid need for it to be implemented by plugin
     */
    void Infer(const Blob &input, Blob &result) override {
        const BlobMap inputs = {{_firstInput, std::shared_ptr<Blob>(const_cast<Blob *>(&input), [](Blob *ptr) {})}};
        BlobMap results = {{_firstOutput, std::shared_ptr<Blob>(&result, [](Blob *ptr) {})}};
        return Infer(inputs, results);
    }

    /**
     * Given optional implementation of deprecated infer to avoid need for it to be implemented by plugin
     */
    void Infer(const BlobMap &input, BlobMap &result) override {
        if (_createdInferRequest == nullptr) {
            THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
        }
        ResponseDesc resp;
        StatusCode sts;

        auto setBlobs = [&](const BlobMap &blobMap) {
            for (auto pair : blobMap) {
                auto blobName = pair.first;
                auto blobPtr = pair.second;
                sts = _createdInferRequest->SetBlob(blobName.c_str(), blobPtr, &resp);
                if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
            }
        };
        setBlobs(input);
        setBlobs(result);

        sts = _createdInferRequest->Infer(&resp);
        if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
    }

    /**
     * Given optional implementation of deprecated infer to avoid need for it to be implemented by plugin
     */
    void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) override {
        if (_createdInferRequest == nullptr) {
            THROW_IE_EXCEPTION << NETWORK_NOT_LOADED_str;
        }
        ResponseDesc resp;
        StatusCode sts = _createdInferRequest->GetPerformanceCounts(perfMap, &resp);
        if (sts != OK) THROW_IE_EXCEPTION << resp.msg;
    }

    /**
     * Given optional implementation of ImportNetwork to avoid need for it to be implemented by plugin
     */
    IExecutableNetwork::Ptr ImportNetwork(const std::string &modelFileName, const std::map<std::string, std::string> &config) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    /**
     * Given optional implementation of SetConfig to avoid need for it to be implemented by plugin
     */
    void SetConfig(const std::map<std::string, std::string> &config) override {}

    /**
     * Given optional implementation of SetLogCallback to avoid need for it to be implemented by plugin
     */
    void SetLogCallback(IErrorListener &listener) override {}

    /**
     * Given optional implementation of AddExtension to avoid need for it to be implemented by plugin
     */
    void AddExtension(InferenceEngine::IExtensionPtr extension) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
    /**
     * @depricated Use the version with config parameter
     */
    void QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const override {
        QueryNetwork(network, {}, res);
    }

    void QueryNetwork(const ICNNNetwork &network, const std::map<std::string, std::string>& config, QueryNetworkResult &res) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }


protected:
    IExecutableNetwork::Ptr _loadedNetwork;
    std::string _firstInput;
    std::string _firstOutput;
    IInferRequest::Ptr _createdInferRequest;
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    std::map<std::string, std::string> _config;
    bool _isDeprecatedLoad = false;
};

}  // namespace InferenceEngine
