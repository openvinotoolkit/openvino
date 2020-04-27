// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "hetero_executable_network.hpp"
#include "hetero_async_infer_request.hpp"
#include "ie_util_internal.hpp"
#include "hetero_graph_splitter.hpp"
#include "file_utils.h"
#include "xml_parse_utils.h"

#include <vector>
#include <deque>
#include <map>
#include <utility>
#include <fstream>
#include <algorithm>
#include <string>
#include <memory>
#include <unordered_set>
#include <array>
#include <cstdint>

#include "details/caseless.hpp"
#include "ie_plugin_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "cpp_interfaces/base/ie_inference_plugin_api.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include "precision_utils.h"
#include "hetero_plugin.hpp"
#include "network_serializer.h"

using namespace InferenceEngine;
using namespace details;
using namespace HeteroPlugin;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;

namespace {

void forward(const CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers) {
    for (const auto& out : layer->outData) {
        for (const auto& out_link : out->getInputTo()) {
            const auto& nextLayer = out_link.second;
            if (nullptr != nextLayer) {
                layers.emplace_back(nextLayer);
            }
        }
    }
}

template<class T>
void traverse(T& inputs,
              std::function<void(InferenceEngine::CNNLayerPtr& layer)> apply,
              std::function<void(const InferenceEngine::CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers)> expand = forward) {
    std::unordered_set<InferenceEngine::CNNLayerPtr> visitedObjects;
    std::deque<InferenceEngine::CNNLayerPtr>         layersToCheck;

    layersToCheck.insert(layersToCheck.end(), inputs.begin(), inputs.end());

    while (!layersToCheck.empty()) {
        auto& layer = layersToCheck.front();
        if (visitedObjects.insert(layer).second) {
            apply(layer);
            expand(layer, layersToCheck);
        }
        layersToCheck.pop_front();
    }
}

void traverse(InferenceEngine::ICNNNetwork& network,
              std::function<void(InferenceEngine::CNNLayerPtr& layer)> apply,
              std::function<void(const InferenceEngine::CNNLayerPtr& layer,
              std::deque<InferenceEngine::CNNLayerPtr>& layers)> expand = forward) {
    std::vector<InferenceEngine::CNNLayerPtr> layers;

    InferenceEngine::InputsDataMap inputs;
    network.getInputsInfo(inputs);
    for (const auto& input : inputs) {
        const auto data = input.second->getInputData();
        for (const auto& to : data->getInputTo()) {
            const auto nextLayer = to.second;
            assert(nullptr != nextLayer);
            layers.emplace_back(nextLayer);
        }
    }

    traverse(layers, apply, expand);
}

std::vector<std::string> getAffinities(InferenceEngine::ICNNNetwork &network) {
    std::vector<std::string> ret;
    std::unordered_set<std::string> affinities;
    traverse(network,
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

HeteroExecutableNetwork::HeteroExecutableNetwork(const InferenceEngine::ICNNNetwork&  network_,
                                                 const Engine::Configs&         config,
                                                 Engine*                        plugin):
    InferenceEngine::ExecutableNetworkThreadSafeDefault(
        nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _plugin{plugin},
    _name{network_.getName()},
    _config{config} {
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

    auto itDumpDotFile = _config.find(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));
    bool dumpDotFile = itDumpDotFile != _config.end() ? itDumpDotFile->second == YES : false;
#ifndef NDEBUG
    dumpDotFile  = true;
#endif

    if (allEmpty) {
        auto it = _config.find("TARGET_FALLBACK");
        if (it != _config.end()) {
            plugin->SetAffinity(network, _config);
        } else {
            THROW_IE_EXCEPTION << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
        }
    } else {
        if (dumpDotFile) {
            std::unordered_set<std::string> devicesSet;
            details::CNNNetworkIterator i(&network);
            while (i != details::CNNNetworkIterator()) {
                CNNLayer::Ptr layer = *i;
                if (!layer->affinity.empty()) {
                    devicesSet.insert(layer->affinity);
                }
                i++;
            }
            std::vector<std::string> devices{std::begin(devicesSet), std::end(devicesSet)};
            std::stringstream stream(std::stringstream::out);
            stream << "hetero_affinity_" << network.getName() << ".dot";
            std::ofstream file(stream.str().c_str());
            saveGraphToDot(network, file, HeteroLayerColorer{devices});
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


    std::vector<NetworkDesc> descs;
    std::vector<CNNLayerPtr> tempLayers;

    for (auto &&subgraph : subgraphs) {
        assert(!subgraph.empty());
        auto affinity = (*subgraph.begin())->affinity;
        assert(!affinity.empty());
        _affinities.push_back(affinity);
        if (_plugin->_plugins.end() == _plugin->_plugins.find(affinity)) {
            IE_SUPPRESS_DEPRECATED_START
            _plugin->_plugins[affinity] = _plugin->GetDevicePlugin(affinity);
            IE_SUPPRESS_DEPRECATED_END
        }
    }

    if (dumpDotFile) {
        std::stringstream stream(std::stringstream::out);
        stream << "hetero_subgraphs_" << network.getName() << ".dot";

        std::ofstream file(stream.str().c_str());
        dumpGraph(network, subgraphs, file);
    }

    InferenceEngine::ICNNNetworkStats* networkStats = nullptr;
    if (StatusCode::OK != network.getStats(&networkStats, nullptr)) {
        networkStats = nullptr;
    }

    for (auto &&subgraph : subgraphs) {
        auto affinity = (*subgraph.begin())->affinity;
        tempLayers.assign(subgraph.begin(), subgraph.end());
        auto tempNetwork = cloneNet(tempLayers, networkStats);
        auto name = network.getName() + "_" + std::to_string(std::distance(subgraphs.data(), &subgraph));
        tempNetwork->setName(name);
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
        desc._clonedNetwork = CNNNetwork{tempNetwork};

        descs.emplace_back(std::move(desc));
    }

    for (auto &&d : descs) {
        IExecutableNetwork::Ptr ret;

        auto subnetworkInputs = d._clonedNetwork.getInputsInfo();
        bool isInputSubnetwork = (subnetworkInputs.end() != std::find_first_of(
            subnetworkInputs.begin(), subnetworkInputs.end(),
            externalInputsData.begin(), externalInputsData.end(),
            [] (const InputsDataMap::value_type& lhs, const InputsDataMap::value_type& rhs) {
                return lhs.first == rhs.first;
            }));

        auto cfg = _config;
        cfg[PluginConfigInternalParams::KEY_SUBNETWORK_WITH_NETWORK_INPUTS] = isInputSubnetwork
                                                                              ? CONFIG_VALUE(YES)
                                                                              : CONFIG_VALUE(NO);
        IE_SUPPRESS_DEPRECATED_START
        auto plugin = _plugin->_plugins[d._device];
        d._network = plugin._ref.LoadNetwork(d._clonedNetwork, Engine::GetSupportedConfig(plugin._config, cfg, plugin._ref));
        IE_SUPPRESS_DEPRECATED_END
    }

    networks = std::move(descs);
}

namespace  {

IE_SUPPRESS_DEPRECATED_START
IInferencePluginAPI * getInferencePluginAPIInterface(IInferencePlugin * iplugin) {
    return dynamic_cast<IInferencePluginAPI *>(iplugin);
}

IInferencePluginAPI * getInferencePluginAPIInterface(InferenceEnginePluginPtr iplugin) {
    return getInferencePluginAPIInterface(static_cast<IInferencePlugin *>(iplugin.operator->()));
}
IE_SUPPRESS_DEPRECATED_END

}  // namespace

HeteroExecutableNetwork::HeteroExecutableNetwork(std::istream&                               heteroModel,
                                                 const std::map<std::string, std::string>&   configs,
                                                 Engine*                                     plugin) :
    _plugin(plugin) {
    std::string heteroXmlStr;
    std::getline(heteroModel, heteroXmlStr);

    pugi::xml_document heteroXmlDoc;
    pugi::xml_parse_result res = heteroXmlDoc.load(heteroXmlStr.c_str());

    if (res.status != pugi::status_ok) {
        THROW_IE_EXCEPTION << "Error reading HETERO plugin xml header";
    }

    using namespace XMLParseUtils;

    pugi::xml_node heteroNode = heteroXmlDoc.document_element();

    std::unordered_set<std::string> networkInputs;
    pugi::xml_node inputsNode = heteroNode.child("inputs");
    for (auto inputNode = inputsNode.child("input"); !inputNode.empty();
            inputNode = inputNode.next_sibling("input")) {
        networkInputs.insert(GetStrAttr(inputNode, "name"));
    }

    std::unordered_set<std::string> networkOutputs;
    pugi::xml_node outputsNode = heteroNode.child("outputs");
    for (auto outputNode = outputsNode.child("output"); !outputNode.empty();
            outputNode = outputNode.next_sibling("output")) {
        networkOutputs.insert(GetStrAttr(outputNode, "name"));
    }

    Engine::Configs importedConfigs;
    auto configsNode = heteroNode.child("configs");
    for (auto configNode = configsNode.child("config"); !configNode.empty();
            configNode = configNode.next_sibling("config")) {
            importedConfigs.emplace(GetStrAttr(configNode, "key"), GetStrAttr(configNode, "value"));
    }

    for (auto&& config : configs) {
        importedConfigs[config.first] = config.second;
    }

    std::vector<NetworkDesc> descs;
    pugi::xml_node subnetworksNode = heteroNode.child("subnetworks");
    for (auto subnetworkNode = subnetworksNode.child("subnetwork"); !subnetworkNode.empty();
            subnetworkNode = subnetworkNode.next_sibling("subnetwork")) {
        auto device = GetStrAttr(subnetworkNode, "device");
        _affinities.push_back(device);

        if (_plugin->_plugins.end() == _plugin->_plugins.find(device)) {
            IE_SUPPRESS_DEPRECATED_START
            _plugin->_plugins[device] = _plugin->GetDevicePlugin(device);
            IE_SUPPRESS_DEPRECATED_END
        }

        auto& plugin = _plugin->_plugins[device];
        auto supportedConfig = Engine::GetSupportedConfig(plugin._config, importedConfigs, plugin._ref);
        IE_SUPPRESS_DEPRECATED_START
        auto pluginAPI = getInferencePluginAPIInterface(plugin._ref);
        IE_SUPPRESS_DEPRECATED_END

        InferenceEngine::ExecutableNetwork executableNetwork;
        CNNNetwork cnnnetwork;
        bool loaded = false;
        try {
            executableNetwork = pluginAPI->ImportNetwork(heteroModel, supportedConfig);
        } catch(InferenceEngine::details::InferenceEngineException& ie_ex) {
            if (std::string::npos != std::string{ie_ex.what()}.find(NOT_IMPLEMENTED_str)) {
                // read XML content
                std::string xmlString;
                std::getline(heteroModel, xmlString);
                std::uint64_t dataSize = 0;
                heteroModel.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));

                // read blob content
                InferenceEngine::Blob::Ptr dataBlob;
                if (0 != dataSize) {
                    dataBlob = InferenceEngine::make_shared_blob<std::uint8_t>(
                        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8,
                                                    {static_cast<std::size_t>(dataSize)},
                                                    InferenceEngine::Layout::C));
                    dataBlob->allocate();
                    heteroModel.read(dataBlob->buffer(), dataSize);
                }

                cnnnetwork = _plugin->GetCore()->ReadNetwork(xmlString, std::move(dataBlob));
                auto inputs = cnnnetwork.getInputsInfo();
                auto inputsNode = subnetworkNode.child("inputs");
                for (auto inputNode = inputsNode.child("input"); !inputNode.empty(); inputNode = inputNode.next_sibling("input")) {
                    auto inputName = GetStrAttr(inputNode, "name");
                    inputs[inputName]->setPrecision(Precision::FromStr(GetStrAttr(inputNode, "precision")));
                }

                auto outputsNode = subnetworkNode.child("outputs");
                for (auto outputNode = outputsNode.child("output"); !outputNode.empty(); outputNode = outputNode.next_sibling("output")) {
                    cnnnetwork.addOutput(GetStrAttr(outputNode, "creatorName"), GetUInt64Attr(outputNode, "index"));
                }
                auto outputs = cnnnetwork.getOutputsInfo();
                for (auto outputNode = outputsNode.child("output"); !outputNode.empty(); outputNode = outputNode.next_sibling("output")) {
                    outputs[GetStrAttr(outputNode, "name")]->setPrecision(Precision::FromStr(GetStrAttr(outputNode, "precision")));
                }
                IE_SUPPRESS_DEPRECATED_START
                executableNetwork = plugin._ref.LoadNetwork(cnnnetwork, supportedConfig);
                IE_SUPPRESS_DEPRECATED_END
                loaded = true;
            } else {
                throw;
            }
        }

        for (auto&& input : executableNetwork.GetInputsInfo()) {
            if (networkInputs.end() != networkInputs.find(input.first)) {
                _networkInputs.emplace(input.first, std::const_pointer_cast<InputInfo>(input.second));
            }
        }

        for (auto&& output : executableNetwork.GetOutputsInfo()) {
            if (networkOutputs.end() != networkOutputs.find(output.first)) {
                _networkOutputs.emplace(output.first, std::const_pointer_cast<Data>(output.second));
            }
        }

        descs.emplace_back(NetworkDesc{
            device,
            loaded ? CNNNetwork{cloneNet(static_cast<InferenceEngine::ICNNNetwork&>(cnnnetwork))} : CNNNetwork{},
            executableNetwork,
        });
    }

    networks = std::move(descs);
}

void HeteroExecutableNetwork::ExportImpl(std::ostream& heteroModel) {
    pugi::xml_document doc;
    auto heteroNode = doc.append_child("hetero");
    heteroNode.append_attribute("name").set_value(_name.c_str());

    auto inputsNode = heteroNode.append_child("inputs");
    for (auto&& networkInput : _networkInputs) {
        inputsNode.append_child("input").append_attribute("name").set_value(networkInput.first.c_str());
    }

    auto outputsNode = heteroNode.append_child("outputs");
    for (auto&& networkInput : _networkOutputs) {
        outputsNode.append_child("output").append_attribute("name").set_value(networkInput.first.c_str());
    }

    auto subnetworksNode = heteroNode.append_child("subnetworks");
    for (auto&& subnetwork : networks) {
        auto subnetworkNode = subnetworksNode.append_child("subnetwork");
        subnetworkNode.append_attribute("device").set_value(subnetwork._device.c_str());
        auto subnetworkInputsNode = subnetworkNode.append_child("inputs");
        auto inputInfo = subnetwork._clonedNetwork.getInputsInfo();
        for (auto&& input : inputInfo) {
            auto inputNode = subnetworkInputsNode.append_child("input");
            inputNode.append_attribute("name").set_value(input.first.c_str());
            inputNode.append_attribute("precision").set_value(input.second->getPrecision().name());
        }
        auto subnetworkOutputsNode = subnetworkNode.append_child("outputs");
        auto outputInfo = subnetwork._clonedNetwork.getOutputsInfo();
        for (auto&& output : outputInfo) {
            auto outputNode = subnetworkOutputsNode.append_child("output");
            auto creator = output.second->getCreatorLayer().lock();
            outputNode.append_attribute("creatorName").set_value(creator->name.c_str());
            outputNode.append_attribute("name").set_value(output.first.c_str());
            outputNode.append_attribute("precision").set_value(output.second->getPrecision().name());
            auto& outDatas = creator->outData;
            auto itData = std::find_if(std::begin(outDatas), std::end(outDatas), [&] (const DataPtr& data) {
                return  output.first == data->getName();
            });
            IE_ASSERT(outDatas.end() != itData);
            std::uint64_t index = std::distance(std::begin(outDatas), itData);
            outputNode.append_attribute("index").set_value(std::to_string(index).c_str());
        }
    }

    auto configsNode = heteroNode.append_child("configs");
    for (auto&& config : _config) {
        auto configMode = configsNode.append_child("config");
        configMode.append_attribute("key").set_value(config.first.c_str());
        configMode.append_attribute("value").set_value(config.second.c_str());
    }

    doc.save(heteroModel, nullptr, pugi::format_raw);
    heteroModel << std::endl;

    for (auto&& subnetwork : networks) {
        try {
            subnetwork._network.Export(heteroModel);
        } catch(InferenceEngine::details::InferenceEngineException& ie_ex) {
            if (std::string::npos != std::string{ie_ex.what()}.find(NOT_IMPLEMENTED_str)) {
                pugi::xml_document doc;
                auto dataSize = static_cast<std::uint64_t>(InferenceEngine::Serialization::FillXmlDoc(subnetwork._clonedNetwork, doc));
                doc.save(heteroModel, nullptr, pugi::format_raw);
                heteroModel << std::endl;
                heteroModel.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
                InferenceEngine::Serialization::SerializeBlobs(heteroModel, subnetwork._clonedNetwork);
            } else {
                throw;
            }
        }
    }
}

InferRequestInternal::Ptr HeteroExecutableNetwork::CreateInferRequestImpl(
        InputsDataMap networkInputs,
        OutputsDataMap networkOutputs) {
    HeteroInferRequest::SubRequestsList inferRequests;
    int index = 0;
    for (auto&& subnetwork : networks) {
        HeteroInferRequest::SubRequestDesc desc;
        desc._network = subnetwork._network;
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
    auto asyncTreadSafeImpl = std::make_shared<HeteroAsyncInferRequest>(heteroInferRequest, _taskExecutor, _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<HeteroAsyncInferRequest>(asyncTreadSafeImpl),
                       [](IInferRequest *p) { p->Release(); });
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}

void HeteroExecutableNetwork::GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *) const {
    if (name == "TARGET_FALLBACK") {
        auto it = _config.find(name);
        if (it != _config.end()) {
            result = it->second;
        } else {
            result = std::string{};
        }
    } else if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT) ||
               name == CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)) {
        auto it = _config.find(name);
        IE_ASSERT(it != _config.end());
        result = it->second == YES ? true : false;
    } else {
        // find config key among plugin config keys
        for (auto&& desc : networks) {
            auto execNetwork = desc._network;
            auto param = execNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            for (auto && configKey : param.as<std::vector<std::string>>()) {
                if (configKey == name) {
                    result = execNetwork.GetConfig(configKey);
                    return;
                }
            }
        }

        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}

using Metrics = std::map<std::string, Parameter>;

namespace {

void collectPluginMetrics(std::vector<std::string> & baseMetrics,
                          const std::vector<::Metrics> pluginMetrics) {
    // check whether the metric has unique name and value among all the plugins
    auto isMetricValueUnique = [&](const std::string & key,
                                    const Parameter & value) -> bool {
        if (std::find(baseMetrics.begin(), baseMetrics.end(), key) !=  baseMetrics.end())
            return false;

        for (auto && metrics : pluginMetrics) {
            for (auto && metric : metrics)
                if (key == metric.first && value != metric.second)
                    return false;
        }

        return true;
    };

    // collect only unique metrics
    std::vector<std::string> uniqueMetrics;
    for (auto && metrics : pluginMetrics) {
        for (auto && metric : metrics) {
            if (isMetricValueUnique(metric.first, metric.second)) {
                uniqueMetrics.push_back(metric.first);
            }
        }
    }

    // add plugin specific metrics which don't conflict with base ones
    std::copy(uniqueMetrics.begin(), uniqueMetrics.end(), std::back_inserter(baseMetrics));
}

}  // namespace

void HeteroExecutableNetwork::GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> heteroMetrics = {
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)
        };

        {
            std::vector<::Metrics> pluginMetrics;
            for (auto&& desc : networks) {
                auto execNetwork = desc._network;
                auto param = execNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS));
                ::Metrics metrics;
                for (auto && metricName : param.as<std::vector<std::string>>()) {
                    metrics[metricName] = execNetwork.GetMetric(metricName);
                }
                pluginMetrics.push_back(std::move(metrics));
            }

            collectPluginMetrics(heteroMetrics, pluginMetrics);
        }

        result = IE_SET_METRIC(SUPPORTED_METRICS, heteroMetrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> heteroConfigKeys = {
            "TARGET_FALLBACK",
            HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)
        };

        {
            std::vector<::Metrics> pluginConfigKeys;
            for (auto&& desc : networks) {
                auto execNetwork = desc._network;
                auto param = execNetwork.GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                ::Metrics configKeys;
                for (auto && metricName : param.as<std::vector<std::string>>()) {
                    configKeys[metricName] = execNetwork.GetConfig(metricName);
                }
                pluginConfigKeys.push_back(std::move(configKeys));
            }

            collectPluginMetrics(heteroConfigKeys, pluginConfigKeys);
        }

        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, heteroConfigKeys);
    } else if (METRIC_KEY(NETWORK_NAME) == name) {
        result = IE_SET_METRIC(NETWORK_NAME, _name);
    } else if (METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        unsigned int value = 0u;
        for (auto&& desc : networks) {
            value = std::max(value, desc._network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>());
        }
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        // find metric key among plugin metrics
        for (auto&& desc : networks) {
            auto execNetwork = desc._network;
            auto param = execNetwork.GetMetric(METRIC_KEY(SUPPORTED_METRICS));
            for (auto && metricKey : param.as<std::vector<std::string>>()) {
                if (metricKey == name) {
                    result = execNetwork.GetMetric(metricKey);
                    return;
                }
            }
        }

        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}
