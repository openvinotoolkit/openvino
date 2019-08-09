// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "hetero_executable_network.hpp"
#include "hetero_async_infer_request.hpp"
#include "ie_util_internal.hpp"
#include "hetero_device_loader.hpp"

#include <vector>
#include <map>
#include <utility>
#include <fstream>
#include <algorithm>
#include <string>
#include <memory>
#include <unordered_set>
#include <array>

#include <ie_plugin_dispatcher.hpp>
#include <ie_graph_splitter.hpp>
#include "fallback_policy.hpp"
#include "details/caseless.hpp"
#include "ie_plugin_config.hpp"
#include "cpp_interfaces/base/ie_inference_plugin_api.hpp"
#include "cpp_interfaces/impl/ie_plugin_internal.hpp"
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
    static const std::array<const char *, 9> colors{{"#FFC405",
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
                                                 const ICore * core,
                                                 const std::map<std::string, std::string> &config,
                                                 const std::vector<InferenceEngine::IExtensionPtr> &extensions,
                                                 MapDeviceLoaders& deviceLoaders,
                                                 InferenceEngine::IErrorListener *listener) :
    _deviceLoaders(deviceLoaders),
    _name(network.getName()) {
    load(network, core, _config = config, extensions, listener);
}

void dla_layer_colorer(const CNNLayerPtr layer,
                       ordered_properties &printed_properties,
                       ordered_properties &node_properties);

void HeteroExecutableNetwork::load(InferenceEngine::ICNNNetwork &network_,
                                   const ICore * core,
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
        FallbackPolicy fbPolicy(_deviceLoaders, dumpDotFile, core);
        auto it = config.find("TARGET_FALLBACK");
        if (it != config.end()) {
            fbPolicy.init(it->second, config, extensions);
            IE_SUPPRESS_DEPRECATED_START
            if (listener)
                for (auto& device_loader : _deviceLoaders)
                    device_loader.second->SetLogCallback(*listener);
            IE_SUPPRESS_DEPRECATED_END
            fbPolicy.setAffinity(fbPolicy.getAffinities(config, network), network);
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
            break;
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

    // we need to create plugins first to use them later during selection of best precision for intermediate blobs
    for (auto &&subgraph : subgraphs) {
        assert(!subgraph.empty());
        auto affinity = (*subgraph.begin())->affinity;
        assert(!affinity.empty());
        _affinities.push_back(affinity);
        if (_deviceLoaders.find(affinity) == _deviceLoaders.end()) {
            // TODO: here is a duplication of the code with FallbackPolicy::init
            IE_SUPPRESS_DEPRECATED_START
            IHeteroDeviceLoader::Ptr loader;
            loader = std::make_shared<HeteroDeviceLoader>(affinity, core);
            HeteroDeviceLoader *pdl = static_cast<HeteroDeviceLoader *>(loader.get());
            IE_SUPPRESS_DEPRECATED_END
            pdl->initConfigs(config, extensions);
            _deviceLoaders[affinity] = loader;
        }
        IE_SUPPRESS_DEPRECATED_START
        if (listener)
            _deviceLoaders[affinity]->SetLogCallback(*listener);
        IE_SUPPRESS_DEPRECATED_END
    }

    InferenceEngine::ICNNNetworkStats* networkStats = nullptr;
    if (StatusCode::OK != network.getStats(&networkStats, nullptr)) {
        networkStats = nullptr;
    }


    for (auto &&subgraph : subgraphs) {
        auto affinity = (*subgraph.begin())->affinity;
        tempLayers.assign(subgraph.begin(), subgraph.end());
        auto tempNetwork = cloneNet(tempLayers, networkStats);
        tempNetwork->setName(network.getName() + "_" + std::to_string(std::distance(subgraphs.data(), &subgraph)));
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
                inp->second->setPrecision(it.second->getPrecision());
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
                it.second->setPrecision(Precision::FP32);
            }
        }

        OutputsDataMap tmpOutputs;
        tempNetwork->getOutputsInfo(tmpOutputs);
        for (auto &&o : tmpOutputs) {
            if (externalOutputsData.find(o.first) == externalOutputsData.end()) {
                o.second->setPrecision(Precision::FP32);
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
        IE_SUPPRESS_DEPRECATED_START
        StatusCode status = d._deviceLoader->LoadNetwork(d._device, ret, *d._clonedNetwork, config, &resp);
        IE_SUPPRESS_DEPRECATED_END
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

void HeteroExecutableNetwork::GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *) const {
    if (name == "TARGET_FALLBACK") {
        auto it = _config.find(name);
        IE_ASSERT(it != _config.end());
        result = it->second;
    } else if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT) ||
               name == CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)) {
        auto it = _config.find(name);
        IE_ASSERT(it != _config.end());
        result = it->second == YES ? true : false;
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}

void HeteroExecutableNetwork::GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        result = IE_SET_METRIC(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{
            "TARGET_FALLBACK",
            HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)});
    } else if (METRIC_KEY(NETWORK_NAME) == name) {
        result = IE_SET_METRIC(NETWORK_NAME, _name);
    } else if (METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        unsigned int value = 0u;
        for (auto&& desc : networks) {
            value = std::max(value, desc.network->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>());
        }
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}
