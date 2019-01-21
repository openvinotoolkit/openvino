// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero_executable_network.h"
#include "hetero_async_infer_request.h"
#include "ie_util_internal.hpp"
#include "hetero_device_loader.h"

#include <array>
#include <set>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <algorithm>

#include <ie_plugin_dispatcher.hpp>
#include <ie_graph_splitter.hpp>
#include "fallback_policy.h"
#include "details/caseless.hpp"
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include "precision_utils.h"

using namespace InferenceEngine;
using namespace details;
using namespace HeteroPlugin;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;

namespace {
std::vector<std::string> getAffinities(InferenceEngine::ICNNNetwork &network) {
    std::vector<std::string> ret;
    std::unordered_set<std::string> affinities;
    traverse::traverse(network,
                       [&](const InferenceEngine::CNNLayerPtr &layer) {
                           assert(nullptr != layer);
                           if (!contains(affinities, layer->affinity)) {
                               affinities.insert(layer->affinity);
                               ret.push_back(layer->affinity);
                           }
                       });
    return ret;
}

void dumpGraph(InferenceEngine::ICNNNetwork &network,
               const std::vector<LayersSet> &subgraphs,
               std::ostream &stream) {
    static const std::array<const char *, 9> colors{{
                                                            "#FFC405",
                                                            "#20F608",
                                                            "#F1F290",
                                                            "#C405FF",
                                                            "#BCFF05",
                                                            "#05FFC4",
                                                            "#FFC405",
                                                            "#5A5DF0",
                                                            "#FF2E05"}};
    auto split_color = [subgraphs](const CNNLayerPtr layer,
                                   ordered_properties &printed_properties,
                                   ordered_properties &node_properties) {
        for (size_t i = 0; i < subgraphs.size(); i++) {
            for (auto s : subgraphs[i]) {
                if (s->name == layer->name) {
                    node_properties.emplace_back(
                            "fillcolor",
                            colors[std::min(i, colors.size() - 1)]);
                    printed_properties.insert(printed_properties.begin(),
                                              std::pair<std::string, std::string>("subgraph#", std::to_string(i)));
                    printed_properties.insert(printed_properties.begin(),
                                              std::pair<std::string, std::string>("device", layer->affinity));
                    return;
                }
            }
        }
    };

    saveGraphToDot(network, stream, split_color);
}

}   // namespace

HeteroExecutableNetwork::HeteroExecutableNetwork(InferenceEngine::ICNNNetwork &network,
                                                 const std::map<std::string, std::string> &config,
                                                 const std::vector<InferenceEngine::IExtensionPtr> &extensions,
                                                 MapDeviceLoaders& deviceLoaders,
                                                 InferenceEngine::IErrorListener *listener) :
    _deviceLoaders(deviceLoaders) {
    load(network, config, extensions, listener);
}

void dla_layer_colorer(const CNNLayerPtr layer,
                       ordered_properties &printed_properties,
                       ordered_properties &node_properties);

void HeteroExecutableNetwork::load(InferenceEngine::ICNNNetwork &network_,
                                   const std::map<std::string, std::string> &config,
                                   const std::vector<InferenceEngine::IExtensionPtr> &extensions,
                                   InferenceEngine::IErrorListener *listener) {
    auto networkPtr = cloneNet(network_);
    auto& network = *networkPtr;

    // going over all network, if all layers are not assigned to devices, apply the default fallback policy
    details::CNNNetworkIterator i(&network);
    bool allEmpty = true;
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        if (!layer->affinity.empty()) {
            allEmpty = false;
            break;
        }
        i++;
    }

    auto itDumpDotFile = config.find(KEY_HETERO_DUMP_GRAPH_DOT);
    bool dumpDotFile = itDumpDotFile != config.end() ? itDumpDotFile->second == YES : false;
#ifndef NDEBUG
    dumpDotFile  = true;
#endif

    if (allEmpty) {
        FallbackPolicy fbPolicy(_deviceLoaders, dumpDotFile);
        auto it = config.find("TARGET_FALLBACK");
        if (it != config.end()) {
            fbPolicy.init(it->second, config, extensions);
            if (listener)
                for (auto& device_loader : _deviceLoaders)
                    device_loader.second->SetLogCallback(*listener);
            fbPolicy.setAffinity(config, network);
        } else {
            THROW_IE_EXCEPTION << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
        }
    } else {
        if (dumpDotFile) {
            std::stringstream stream(std::stringstream::out);
            stream << "hetero_affinity_" << network.getName() << ".dot";

            std::ofstream file(stream.str().c_str());
            saveGraphToDot(network, file, dla_layer_colorer);
        }
    }

    details::CNNNetworkIterator el(&network);
    bool someEmptyAffinity = false;
    CNNLayer::Ptr layerEmptyAffinity = nullptr;
    while (el != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *el;
        if (!CaselessEq<std::string>()(layer->type, "input") &&
            layer->affinity.empty()) {
            someEmptyAffinity = true;
            layerEmptyAffinity = layer;
        }
        el++;
    }

    if (allEmpty && someEmptyAffinity) {
        THROW_IE_EXCEPTION << "Hetero plugin used default fallback policy, but some layers eg: \n(Name:" <<
            layerEmptyAffinity->name << ", Type: " << layerEmptyAffinity->type <<
            ") were not able to be assigned on any pointed device.\n" <<
            "It happened because these layers are not supported in plugins by default.\n" <<
            "You need to implement custom layers to support them.";
    } else if (someEmptyAffinity) {
        THROW_IE_EXCEPTION << "Network passed to LoadNetwork has affinity assigned, but some layers eg: \n(Name:" <<
            layerEmptyAffinity->name << ", Type: " << layerEmptyAffinity->type <<
            ") were not assigned to any device.\n" <<
            "It might happen if you assigned layers amnually and missed some layers or\n" <<
            "if you used some automatic assigning mode which decided that these layers are not\n" <<
            "supported by any plugin";
    }


    InputsDataMap externalInputsData;
    network.getInputsInfo(externalInputsData);

    OutputsDataMap externalOutputsData;
    network.getOutputsInfo(externalOutputsData);

    auto subgraphs = splitGraph(network, getAffinities(network));

    sortSubgraphs(subgraphs);

    if (dumpDotFile) {
        std::stringstream stream(std::stringstream::out);
        stream << "hetero_subgraphs_" << network.getName() << ".dot";

        std::ofstream file(stream.str().c_str());
        dumpGraph(network, subgraphs, file);
    }

    std::vector<NetworkDesc> descs;
    PluginDispatcher dispatcher({ "" });
    std::vector<CNNLayerPtr> tempLayers;

    // we need to create plugins first to use them later during selection of best precisino for intermediate blobs
    for (auto &&subgraph : subgraphs) {
        assert(!subgraph.empty());
        auto affinity = (*subgraph.begin())->affinity;
        assert(!affinity.empty());
        if (_deviceLoaders.find(affinity) == _deviceLoaders.end()) {
            // TODO: here is a duplication of the code with FallbackPolicy::init
            IHeteroDeviceLoader::Ptr loader;
            loader = std::make_shared<HeteroDeviceLoader>(affinity);
            HeteroDeviceLoader *pdl = dynamic_cast<HeteroDeviceLoader *>(loader.get());
            pdl->initConfigs(config, extensions);
            _deviceLoaders[affinity] = loader;
        }
        if (listener)
            _deviceLoaders[affinity]->SetLogCallback(*listener);
    }

    for (auto &&subgraph : subgraphs) {
        auto affinity = (*subgraph.begin())->affinity;
        tempLayers.assign(subgraph.begin(), subgraph.end());
        auto tempNetwork = cloneNet(tempLayers);
        // restoring some outputs from original net if they are not marked as output automatically
        // this might happen if output was set manually for origin network and
        // it doesn't go to next subgraph
        for (auto il : tempLayers) {
            if (externalOutputsData.find(il->name) != externalOutputsData.end()) {
                tempNetwork->addOutput(il->name);
            }
        }

        tempNetwork->setPrecision(network.getPrecision());

        // update of pre-processing info
        InputsDataMap clonedInputs;
        tempNetwork->getInputsInfo(clonedInputs);
        for (auto &&it : externalInputsData) {
            auto inp = clonedInputs.find(it.first);
            if (inp != clonedInputs.end() && nullptr != inp->second) {
                inp->second->setInputPrecision(it.second->getInputPrecision());
                inp->second->getPreProcess() = it.second->getPreProcess();
            }
        }
        // go over all inputs/outputs and right now
        // set precision for intermediate data (not for external) to FP32
        // later on we have to add Plugin::getPreferableInputPrecision(network) and
        // Plugin::getPreferableOutputPrecision(network) and set precision based on this info
        // TODO(amalyshe) add clever selectino of precision for intermediate blobs
        for (auto &&it : clonedInputs) {
            if (externalInputsData.find(it.first) == externalInputsData.end()) {
                it.second->setInputPrecision(Precision::FP32);
            }
        }

        OutputsDataMap tmpOutputs;
        tempNetwork->getOutputsInfo(tmpOutputs);
        for (auto &&o : tmpOutputs) {
            if (externalOutputsData.find(o.first) == externalOutputsData.end()) {
                o.second->setPrecision(Precision::FP32);
            }
        }

        // Temporal solution until each plugin starts to support desirable precision
        // Only for CPU registered device we are changing all FP16 types to FP32 and convert blobs if any
        // TODO(amalyshe) remove this hack to preoper network.setPrecision(FP16) and feeding to CPU plugin
        if (affinity == "CPU") {
            tempNetwork->setPrecision(Precision::FP32);
            details::CNNNetworkIterator itcpu(reinterpret_cast<ICNNNetwork *>(tempNetwork.get()));
            bool allEmpty = true;
            while (itcpu != details::CNNNetworkIterator()) {
                CNNLayer::Ptr layer = *itcpu;
                layer->precision = Precision::FP32;
                // take all input and output data, set FP32 precision for them
                for (auto o : layer->outData) {
                    if (externalInputsData.find(o->getName()) == externalInputsData.end() &&
                        externalOutputsData.find(o->getName()) == externalOutputsData.end()) {
                        o->setPrecision(Precision::FP32);
                    }
                }
                for (auto i : layer->insData) {
                    if (externalInputsData.find(i.lock()->getName()) == externalInputsData.end() &&
                        externalOutputsData.find(i.lock()->getName()) == externalOutputsData.end()) {
                        i.lock()->setPrecision(Precision::FP32);
                    }
                }

                auto convertBlobFP16toFP32 = [](Blob::Ptr blob) -> Blob::Ptr {
                    Blob::Ptr weightsBlob = make_shared_blob<float>(Precision::FP32, blob->layout(), blob->dims());
                    weightsBlob->allocate();
                    float* target = weightsBlob->buffer().as<float*>();
                    short* source = blob->buffer().as<short *>();
                    PrecisionUtils::f16tof32Arrays(target, source, blob->size(), 1.0f, 0.0f);
                    return weightsBlob;
                };
                // convert blobs
                auto wLayer = dynamic_cast<InferenceEngine::WeightableLayer *>(layer.get());
                if (wLayer) {
                    // verify
                    if (wLayer->_weights && wLayer->_weights->precision() == Precision::FP16) {
                        wLayer->_weights = convertBlobFP16toFP32(wLayer->_weights);
                    } else if (wLayer->_weights && wLayer->_weights->precision() != Precision::FP32) {
                        THROW_IE_EXCEPTION << "weights for layer '" << wLayer->name << "' has unsupported precision";
                    }
                    if (wLayer->_biases && wLayer->_biases->precision() == Precision::FP16) {
                        wLayer->_biases = convertBlobFP16toFP32(wLayer->_biases);
                    } else if (wLayer->_biases && wLayer->_biases->precision() != Precision::FP32) {
                        THROW_IE_EXCEPTION << "biases for layer '" << wLayer->name << "' has unsupported precision";
                    }
                }
                for (auto&& blob : layer->blobs) {
                    auto&& data = blob.second;
                    if (nullptr != data) {
                        if (data->precision() == Precision::FP16) {
                            data = convertBlobFP16toFP32(data);
                        } else if (data->precision() != Precision::FP32) {
                            THROW_IE_EXCEPTION << "weights '" << blob.first << "' for layer '" << layer->name << "' has unsupported precision";
                        }  // else no need to convert
                    }
                }
                itcpu++;
            }
        }

        NetworkDesc desc;
        desc._device = affinity;
        desc._deviceLoader = _deviceLoaders[affinity];

        desc._clonedNetwork = tempNetwork;
        InputsDataMap inputs;
        desc._clonedNetwork->getInputsInfo(inputs);
        for (auto i : inputs) {
            desc._iNames.insert(i.first);
        }
        OutputsDataMap outputs;
        desc._clonedNetwork->getOutputsInfo(outputs);
        for (auto o : outputs) {
            desc._oNames.insert(o.first);
        }

        descs.emplace_back(std::move(desc));
    }

    for (auto &&d : descs) {
        IExecutableNetwork::Ptr ret;
        ResponseDesc resp;
        StatusCode status = d._deviceLoader->LoadNetwork(d._device, ret, *d._clonedNetwork, config, &resp);
        if (status != OK) {
            THROW_IE_EXCEPTION << resp.msg;
        }
        d.network = std::make_shared<ExecutableNetwork>(ret);
        d._clonedNetwork = nullptr;
    }


    networks = std::move(descs);
}

InferRequestInternal::Ptr HeteroExecutableNetwork::CreateInferRequestImpl(
        InputsDataMap networkInputs,
        OutputsDataMap networkOutputs) {
    HeteroInferRequest::SubRequestsList inferRequests;
    int index = 0;
    for (auto i : networks) {
        HeteroInferRequest::SubRequestDesc desc;
        desc._network = i.network;
        desc._iNames = i._iNames;
        desc._oNames = i._oNames;
        desc._profilingTask = ProfilingTask{"Infer" + std::to_string(index++)};

        inferRequests.push_back(desc);
    }
    return std::make_shared<HeteroInferRequest>(networkInputs,
                                                networkOutputs,
                                                inferRequests);
}

void HeteroExecutableNetwork::CreateInferRequest(IInferRequest::Ptr &asyncRequest) {
    auto heteroInferRequest = std::dynamic_pointer_cast<HeteroInferRequest>(
            CreateInferRequestImpl(_networkInputs, _networkOutputs));
    heteroInferRequest->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncTreadSafeImpl = std::make_shared<HeteroAsyncInferRequest>(
            heteroInferRequest, _taskExecutor, _taskSynchronizer, _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<HeteroAsyncInferRequest>(asyncTreadSafeImpl),
                       [](IInferRequest *p) { p->Release(); });
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}
