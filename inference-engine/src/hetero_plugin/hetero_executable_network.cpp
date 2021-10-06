// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "hetero_executable_network.hpp"
#include "hetero_async_infer_request.hpp"
#include "hetero_itt.hpp"
#include "xml_parse_utils.h"
#include <caseless.hpp>

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

#include "transformations/serialize.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "ie_algorithm.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "hetero_plugin.hpp"
#include <ie_algorithm.hpp>

#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/visualize_tree.hpp>

using namespace InferenceEngine;
using namespace details;
using namespace HeteroPlugin;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;

template<typename T>
using NodeMap = std::unordered_map<ngraph::Node*, T>;

HeteroExecutableNetwork::HeteroExecutableNetwork(const InferenceEngine::CNNNetwork&     network,
                                                 const Engine::Configs&                 config,
                                                 Engine*                                plugin):
    InferenceEngine::ExecutableNetworkThreadSafeDefault(
        nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _heteroPlugin{plugin},
    _name{network.getName()},
    _config{config} {
    auto function = network.getFunction();
    IE_ASSERT(function != nullptr);
    auto clonedFunction = ngraph::clone_function(*function);
    auto itDumpDotFile = _config.find(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));
    bool dumpDotFile = itDumpDotFile != _config.end() ? (itDumpDotFile->second == YES) : false;
#ifndef NDEBUG
    dumpDotFile  = true;
#endif
    QueryNetworkResult queryNetworkResult;
    auto orderedOps = clonedFunction->get_ordered_ops();
    bool allEmpty = true;
    // Get user defined affinity
    for (auto&& node : orderedOps) {
        auto& nodeInfo = node->get_rt_info();
        auto itInfo = nodeInfo.find("affinity");
        if (itInfo != nodeInfo.end()) {
            IE_ASSERT((ngraph::is_type<ngraph::VariantWrapper<std::string>>(itInfo->second)));
            queryNetworkResult.supportedLayersMap.emplace(
                node->get_friendly_name(),
                ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(itInfo->second)->get());
            allEmpty = false;
        }
    }

    if (queryNetworkResult.supportedLayersMap.empty()) {
        auto it = _config.find("TARGET_FALLBACK");
        if (it != _config.end()) {
            queryNetworkResult = _heteroPlugin->QueryNetwork(network, _config);
        } else {
            IE_THROW() << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
        }
    }

    using Input = ngraph::Input<ngraph::Node>;
    using NodeSet = std::unordered_set<ngraph::Node*>;
    using InputSet = std::set<Input>;

    auto InputNode  = [] (const ngraph::Input<ngraph::Node>& input) {
        return input.get_source_output().get_node();
    };

    // Set results, constants and parameters affinity
    for (auto&& node : clonedFunction->get_ops()) {
        if (ngraph::op::is_constant(node) || ngraph::op::is_output(node) || ngraph::op::is_parameter(node)) {
            if (!contains(queryNetworkResult.supportedLayersMap, node->get_friendly_name())) {
                auto& nodeWithAffinityName = ngraph::op::is_output(node)
                                           ? node->input_value(0).get_node()->get_friendly_name()
                                           : node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name();
                auto itAffinity = queryNetworkResult.supportedLayersMap.find(nodeWithAffinityName);
                if (itAffinity == queryNetworkResult.supportedLayersMap.end()) {
                    IE_THROW() << "Node " << nodeWithAffinityName <<
                                        " was not assigned on any pointed device.";
                }
                queryNetworkResult.supportedLayersMap.emplace(node->get_friendly_name(), itAffinity->second);
            }
        }
    }

    std::unordered_set<std::string> devices;
    NodeMap<std::string> affinities;
    // Check that all nodes has user or plugin defined affinities
    for (auto&& node : orderedOps) {
        auto itAffinity = queryNetworkResult.supportedLayersMap.find(node->get_friendly_name());
        if (itAffinity != queryNetworkResult.supportedLayersMap.end()) {
            affinities[node.get()] = itAffinity->second;
            devices.emplace(itAffinity->second);
        } else if (allEmpty) {
            IE_THROW() << "Hetero plugin used default fallback policy, but some layers eg: \n(Name:" <<
                node->get_friendly_name() << ", Type: " << node->get_type_name() <<
                ") were not able to be assigned on any pointed device.\n" <<
                "It happened because these layers are not supported in plugins by default.\n" <<
                "You need to implement custom layers to support them.";
        } else {
            IE_THROW() << "Network passed to LoadNetwork has affinity assigned, but some layers eg: \n(Name:" <<
                node->get_friendly_name() << ", Type: " << node->get_type_name() <<
                ") were not assigned to any device.\n" <<
                "It might happen if you assigned layers manually and missed some layers or\n" <<
                "if you used some automatic assigning mode which decided that these layers are not\n" <<
                "supported by any plugin";
        }
    }

    static const std::array<const char*, 14> colors = {
        "aliceblue",
        "antiquewhite4",
        "aquamarine4",
        "azure4",
        "bisque3",
        "blue1",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk4",
    };

    if (dumpDotFile) {
        ngraph::pass::VisualizeTree{"hetero_affinity_" + _name + ".dot",
            [&] (const ngraph::Node& node, std::vector<std::string>& attributes) {
                auto nodeDevice = queryNetworkResult.supportedLayersMap.at(node.get_friendly_name());
                int colorIndex = 0;
                for (auto&& device : devices) {
                    if (device == nodeDevice) {
                        attributes.push_back(std::string {"fillcolor="} + colors[colorIndex % colors.size()] + " style=filled");
                        auto itLabel = std::find_if(std::begin(attributes), std::end(attributes), [] (const std::string& str) {
                            return str.find("label") != std::string::npos;
                        });
                        auto label = "\\ndevice=" + queryNetworkResult.supportedLayersMap.at(node.get_friendly_name()) + '\"';
                        IE_ASSERT(itLabel != attributes.end());
                        itLabel->pop_back();
                        (*itLabel) += label;
                        break;
                    }
                    colorIndex++;
                }
            }}.run_on_function(ngraph::clone_function(*function));
    }


    NodeMap<InputSet> nodeInputDependencies;
    NodeSet graphInputNodes;
    InputSet subgraphInputs;
    // Get all subgraph inputs using just node affinities. Also collect transitive closure
    for (auto&& node : orderedOps) {
        if (ngraph::op::is_parameter(node) || ngraph::op::is_constant(node)) {
            graphInputNodes.insert(node.get());
            subgraphInputs.insert(Input{node.get(), 0});
            nodeInputDependencies[node.get()].insert(Input{node.get(), 0});
        } else {
            auto inputs = node->inputs();
            auto& nodeInputDependency = nodeInputDependencies[node.get()];
            for (auto&& input : inputs) {
                nodeInputDependency.insert(input);
                auto& inputDependency = nodeInputDependencies[InputNode(input)];
                nodeInputDependency.insert(inputDependency.begin(), inputDependency.end());
                if (affinities[node.get()] != affinities[InputNode(input)]) {
                    subgraphInputs.insert(input);
                }
            }
        }
    }

    // Assign each node subgraph ID
    auto CollectSubgraphs = [&] {
        std::deque<int> subgraphIds;
        NodeMap<int*> subgraphIdPtrs;
        for (auto&& node : orderedOps) {
            auto allNodeInputs = node->inputs();
            std::vector<Input> inputs;
            for (auto&& input : allNodeInputs) {
                if (!contains(subgraphInputs, input)) {
                    inputs.emplace_back(std::move(input));
                }
            }
            if (inputs.empty()) {
                subgraphIds.push_back(subgraphIds.size());
                subgraphIdPtrs.emplace(node.get(), &(subgraphIds.back()));
            } else {
                auto firstInputSubgraphIdPtr = subgraphIdPtrs[InputNode(inputs.front())];
                for (auto&& input : inputs) {
                    auto inputId = *subgraphIdPtrs[InputNode(input)];
                    for (auto& subgraphId : subgraphIds) {
                        if (subgraphId == inputId) {
                            subgraphId = *firstInputSubgraphIdPtr;
                        }
                    }
                }
                subgraphIdPtrs.emplace(node.get(), firstInputSubgraphIdPtr);
            }
        }
        NodeMap<int> result;
        for (auto&& subgraphIdPtr : subgraphIdPtrs) {
            result.emplace(subgraphIdPtr.first, *(subgraphIdPtr.second));
        }
        return result;
    };

    // Split cyclic dependencies.
    for (std::size_t prevSubgraphs = 0, cyclicSplitStep = 0; prevSubgraphs != subgraphInputs.size(); ++cyclicSplitStep) {
        IE_ASSERT(cyclicSplitStep < orderedOps.size());
        prevSubgraphs = subgraphInputs.size();
        auto subgraphIds = CollectSubgraphs();
        // All inputs that belong to the same subgraph as node
        std::unordered_map<ngraph::Node*, InputSet> nodeSubgraphInputDependencies;
        // All inputs that depends on the same subgraph as node
        std::unordered_map<ngraph::Node*, InputSet> nodeSubgraphCyclicInputDependencies;
        for (auto&& node : orderedOps) {
            auto& nodeSubgraphInputDependency = nodeSubgraphInputDependencies[node.get()];
            auto allNodeSubgraphInputs = Intersection(nodeInputDependencies[node.get()], subgraphInputs);
            for (auto&& subgraphInput : allNodeSubgraphInputs) {
                if (subgraphIds[node.get()] == subgraphIds[subgraphInput.get_node()]) {
                    nodeSubgraphInputDependency.emplace(subgraphInput);
                }
            }
            auto& nodeSubgraphCyclicInputDependency = nodeSubgraphCyclicInputDependencies[node.get()];
            for (auto&& subgraphInput : allNodeSubgraphInputs) {
                if (!ngraph::op::is_parameter(subgraphInput.get_node()) &&
                    !ngraph::op::is_constant(subgraphInput.get_node()) &&
                    subgraphIds[node.get()] == subgraphIds[InputNode(subgraphInput)]) {
                    nodeSubgraphCyclicInputDependency.emplace(subgraphInput);
                }
            }
        }

        for (auto&& node : orderedOps) {
            auto& nodeSubgraphCyclicInputDependency = nodeSubgraphCyclicInputDependencies[node.get()];
            if (!nodeSubgraphCyclicInputDependency.empty()) {
                // Collect all subgraph inputs that cyclic subgraph output depends on
                InputSet cyclicInputsDependencies;
                for (auto&& cyclicInput : nodeSubgraphCyclicInputDependency) {
                    for (auto&& input : nodeSubgraphInputDependencies[InputNode(cyclicInput)]) {
                        cyclicInputsDependencies.emplace(input);
                    }
                }
                for (auto&& input : node->inputs()) {
                    auto& inputNodeSubgraphCyclicInputDependency = nodeSubgraphCyclicInputDependencies[InputNode(input)];
                    auto& inputNodeSubgraphInputDependency = nodeSubgraphInputDependencies[InputNode(input)];
                    if (!Intersects(nodeSubgraphCyclicInputDependency,
                                    inputNodeSubgraphCyclicInputDependency) &&
                        Intersects(cyclicInputsDependencies, inputNodeSubgraphInputDependency)) {
                        subgraphInputs.insert(input);
                    }
                }
            }
        }
    }

    auto subgraphIds = CollectSubgraphs();
    // Break graph using insertion of result parameter split
    NodeMap<ngraph::Node*> subgraphParameterToPrevResult;
    std::vector<std::shared_ptr<ngraph::op::Result>> results;
    for (auto&& input : subgraphInputs) {
        if (!ngraph::op::is_parameter(input.get_node()) && !ngraph::op::is_constant(input.get_node())) {
            auto output = input.get_source_output();
            output.remove_target_input(input);
            auto result = std::make_shared<ngraph::op::Result>(output);
            ngraph::copy_runtime_info(output.get_node_shared_ptr(), result);
            auto parameter = std::make_shared<ngraph::op::Parameter>(output.get_element_type(), output.get_partial_shape());
            ngraph::copy_runtime_info(input.get_node()->shared_from_this(), parameter);
            input.replace_source_output(parameter->output(0));
            results.push_back(result);
            subgraphIds.emplace(result.get(), subgraphIds[output.get_node()]);
            subgraphIds.emplace(parameter.get(), subgraphIds[input.get_node()]);
            subgraphParameterToPrevResult.emplace(parameter.get(), result.get());
            _blobNameMap.emplace(parameter->get_friendly_name(),
                                 output.get_node()->get_friendly_name() +
                                 ((output.get_node()->get_output_size() != 1)
                                 ? ("." + std::to_string(output.get_index())) : std::string{}));
        }
    }

    struct Subgraph {
        ngraph::ResultVector    _results;
        ngraph::ParameterVector _parameters;
        ngraph::SinkVector      _sinks;
        std::string             _affinity;
    };
    std::unordered_map<int, Subgraph> subgraphs;
    // Extracts subgraph parameters, results and affinities
    for (auto&& subgraphIdPtrValue : subgraphIds) {
        auto node = subgraphIdPtrValue.first;
        auto& subgraph = subgraphs[subgraphIdPtrValue.second];
        if (ngraph::op::is_output(node)) {
            subgraph._results.emplace_back(
                std::dynamic_pointer_cast<ngraph::op::v0::Result>(node->shared_from_this()));
        } else if (ngraph::op::is_parameter(node)) {
            subgraph._parameters.emplace_back(
                std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(node->shared_from_this()));
        } else if (ngraph::op::is_sink(node)) {
            subgraph._sinks.emplace_back(
                    std::dynamic_pointer_cast<ngraph::op::Sink>(node->shared_from_this()));
        }
        auto itAffinity = affinities.find(node);
        if (itAffinity != affinities.end()) {
            subgraph._affinity = itAffinity->second;
        }
    }

    // Subgraph topological sort
    std::vector<Subgraph> allSubgraphs;
    for (auto&& subgraph : subgraphs) {
        allSubgraphs.emplace_back(std::move(subgraph.second));
    }

    std::vector<Subgraph> orderedSubgraphs;
    NodeSet prevResults;
    size_t subgraphTopoSortsStep = 0;
    do {
        IE_ASSERT(subgraphTopoSortsStep++ < subgraphs.size());
        std::vector<Subgraph> nextSubgraphs;
        auto IsNextSubGraph = [&] (const Subgraph& subgraph) {
            auto& parameters = subgraph._parameters;
            return std::all_of(parameters.begin(), parameters.end(),
                    [&] (const ngraph::ParameterVector::value_type& parameter) {
                    return contains(graphInputNodes, parameter.get()) ||
                           contains(prevResults, subgraphParameterToPrevResult[parameter.get()]);});
        };
        std::remove_copy_if(std::begin(allSubgraphs), std::end(allSubgraphs),
                            std::back_inserter(nextSubgraphs),
                            [&] (const Subgraph& subgraph) { return !IsNextSubGraph(subgraph);});
        allSubgraphs.erase(
            std::remove_if(std::begin(allSubgraphs), std::end(allSubgraphs), IsNextSubGraph),
            std::end(allSubgraphs));
        for (auto&& subgraph :  nextSubgraphs) {
            for (auto&& result : subgraph._results) {
                prevResults.insert(result.get());
            }
        }
        std::move(std::begin(nextSubgraphs), std::end(nextSubgraphs), std::back_inserter(orderedSubgraphs));
    } while (!allSubgraphs.empty());

    InputsDataMap externalInputsData = network.getInputsInfo();
    OutputsDataMap externalOutputsData = network.getOutputsInfo();
    _networks.resize(orderedSubgraphs.size());
    std::vector<std::shared_ptr<ngraph::Function>> subFunctions(orderedSubgraphs.size());
    int id = 0;
    for (auto&& subgraph : orderedSubgraphs) {
        _networks[id]._device = subgraph._affinity;
        subFunctions[id] =
            std::make_shared<ngraph::Function>(subgraph._results, subgraph._sinks, subgraph._parameters,
                                                     _name + '_' + std::to_string(id));
        _networks[id]._clonedNetwork = CNNNetwork{subFunctions[id]};
        // update of pre-processing info
        auto clonedInputs = _networks[id]._clonedNetwork.getInputsInfo();
        for (auto&& externalInput : externalInputsData) {
            auto itClonedInput = clonedInputs.find(externalInput.first);
            if (itClonedInput != clonedInputs.end() && nullptr != itClonedInput->second) {
                itClonedInput->second->getPreProcess() = externalInput.second->getPreProcess();
                itClonedInput->second->setPrecision(externalInput.second->getPrecision());
                itClonedInput->second->setLayout(externalInput.second->getLayout());
            }
        }
        // update output info
        auto clonedOutputs = _networks[id]._clonedNetwork.getOutputsInfo();
        for (auto&& externalOutput : externalOutputsData) {
            auto itClonedOutput = clonedOutputs.find(externalOutput.first);
            if (itClonedOutput != clonedOutputs.end() && nullptr != itClonedOutput->second) {
                itClonedOutput->second->setPrecision(externalOutput.second->getPrecision());
                itClonedOutput->second->setLayout(externalOutput.second->getLayout());
            }
        }
        ++id;
    }
    if (dumpDotFile) {
        ngraph::pass::VisualizeTree{"hetero_subgraphs_" + _name + ".dot",
            [&] (const ngraph::Node& node, std::vector<std::string>& attributes) {
                for (size_t i = 0; i < subFunctions.size(); i++) {
                    for (auto&& nodeInSubfunction : subFunctions[i]->get_ops()) {
                        if (nodeInSubfunction->get_friendly_name() == node.get_friendly_name()) {
                            attributes.push_back(std::string {"fillcolor="} + colors[i % colors.size()] + " style=filled");
                            auto itLabel = std::find_if(std::begin(attributes), std::end(attributes), [] (const std::string& str) {
                                return str.find("label") != std::string::npos;
                            });
                            auto label = "\\nsubgraph=" + std::to_string(i) + "\\n"
                                       + "device=" + queryNetworkResult.supportedLayersMap.at(node.get_friendly_name()) + '\"';
                            IE_ASSERT(itLabel != attributes.end());
                            itLabel->pop_back();
                            (*itLabel) += label;
                        }
                    }
                }
            }}.run_on_function(ngraph::clone_function(*function));
    }
    for (auto&& network : _networks) {
        auto metaDevices = _heteroPlugin->GetDevicePlugins(network._device, _config);
        metaDevices[network._device].emplace(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE), "");
        network._network = _heteroPlugin->GetCore()->LoadNetwork(network._clonedNetwork,
            network._device, metaDevices[network._device]);
    }
}

HeteroExecutableNetwork::HeteroExecutableNetwork(std::istream&                               heteroModel,
                                                 const std::map<std::string, std::string>&   configs,
                                                 Engine*                                     heteroPlugin) :
    _heteroPlugin(heteroPlugin) {
    std::string heteroXmlStr;
    std::getline(heteroModel, heteroXmlStr);

    pugi::xml_document heteroXmlDoc;
    pugi::xml_parse_result res = heteroXmlDoc.load_string(heteroXmlStr.c_str());

    if (res.status != pugi::status_ok) {
        IE_THROW(NetworkNotRead) << "Error reading HETERO plugin xml header";
    }

    using namespace XMLParseUtils;

    pugi::xml_node heteroNode = heteroXmlDoc.document_element();
    _name = GetStrAttr(heteroNode, "name");

    std::unordered_set<std::string> networkInputs;
    pugi::xml_node inputsNode = heteroNode.child("inputs");
    FOREACH_CHILD(inputNode, inputsNode, "input")  {
        networkInputs.insert(GetStrAttr(inputNode, "name"));
    }

    std::unordered_set<std::string> networkOutputs;
    pugi::xml_node outputsNode = heteroNode.child("outputs");
    FOREACH_CHILD(outputNode, outputsNode, "output") {
        networkOutputs.insert(GetStrAttr(outputNode, "name"));
    }

    Engine::Configs importedConfigs;
    auto configsNode = heteroNode.child("configs");
    FOREACH_CHILD(configNode, configsNode, "config") {
        importedConfigs.emplace(GetStrAttr(configNode, "key"), GetStrAttr(configNode, "value"));
    }

    auto blobNamesNode = heteroNode.child("blob_names_map");
    FOREACH_CHILD(blobNameNode, blobNamesNode, "blob_name_map") {
        _blobNameMap.emplace(GetStrAttr(blobNameNode, "key"), GetStrAttr(blobNameNode, "value"));
    }

    for (auto&& config : configs) {
        importedConfigs[config.first] = config.second;
    }

    std::vector<NetworkDesc> descs;
    pugi::xml_node subnetworksNode = heteroNode.child("subnetworks");
    FOREACH_CHILD(subnetworkNode, subnetworksNode, "subnetwork") {
        auto deviceName = GetStrAttr(subnetworkNode, "device");

        auto metaDevices = _heteroPlugin->GetDevicePlugins(deviceName, importedConfigs);
        assert(metaDevices.size() == 1);
        auto& loadConfig = metaDevices[deviceName];

        InferenceEngine::SoExecutableNetworkInternal executableNetwork;
        CNNNetwork cnnnetwork;
        bool loaded = false;
        if (_heteroPlugin->GetCore()->DeviceSupportsImportExport(deviceName)) {
            executableNetwork = _heteroPlugin->GetCore()->ImportNetwork(heteroModel, deviceName, loadConfig);
        } else {
            // read XML content
            std::string xmlString;
            std::uint64_t dataSize = 0;
            heteroModel.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            xmlString.resize(dataSize);
            heteroModel.read(const_cast<char*>(xmlString.c_str()), dataSize);

            // read blob content
            InferenceEngine::Blob::Ptr dataBlob;
            heteroModel.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            if (0 != dataSize) {
                dataBlob = InferenceEngine::make_shared_blob<std::uint8_t>(
                    InferenceEngine::TensorDesc(InferenceEngine::Precision::U8,
                                                {static_cast<std::size_t>(dataSize)},
                                                InferenceEngine::Layout::C));
                dataBlob->allocate();
                heteroModel.read(dataBlob->buffer(), dataSize);
            }

            cnnnetwork = _heteroPlugin->GetCore()->ReadNetwork(xmlString, std::move(dataBlob));
            auto inputs = cnnnetwork.getInputsInfo();
            auto inputsNode = subnetworkNode.child("inputs");
            FOREACH_CHILD(inputNode, inputsNode, "input") {
                auto inputName = GetStrAttr(inputNode, "name");
                inputs[inputName]->setPrecision(Precision::FromStr(GetStrAttr(inputNode, "precision")));
            }

            auto outputs = cnnnetwork.getOutputsInfo();
            auto outputsNode = subnetworkNode.child("outputs");
            FOREACH_CHILD(outputNode, outputsNode, "output") {
                auto outputName = GetStrAttr(outputNode, "name");
                outputs[outputName]->setPrecision(Precision::FromStr(GetStrAttr(outputNode, "precision")));
            }

            executableNetwork = _heteroPlugin->GetCore()->LoadNetwork(cnnnetwork, deviceName, loadConfig);
            loaded = true;
        }

        // restore network inputs and outputs
        for (auto&& input : executableNetwork->GetInputsInfo()) {
            if (networkInputs.end() != networkInputs.find(input.first)) {
                _networkInputs.emplace(input.first, std::make_shared<InputInfo>(*input.second));
            }
        }

        for (auto&& output : executableNetwork->GetOutputsInfo()) {
            if (networkOutputs.end() != networkOutputs.find(output.first)) {
                _networkOutputs.emplace(output.first, std::make_shared<Data>(*output.second));
            }
        }

        descs.emplace_back(NetworkDesc{
            deviceName,
            loaded ? cnnnetwork : CNNNetwork{},
            executableNetwork,
        });
    }

    // save state
    this->_config = importedConfigs;
    this->_networks = std::move(descs);
    this->SetPointerToPlugin(_heteroPlugin->shared_from_this());
}

void HeteroExecutableNetwork::Export(std::ostream& heteroModel) {
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
    for (auto&& subnetwork : _networks) {
        auto subnet = subnetwork._clonedNetwork;
        IE_ASSERT(subnet.getFunction() != nullptr);

        auto subnetworkNode = subnetworksNode.append_child("subnetwork");
        subnetworkNode.append_attribute("device").set_value(subnetwork._device.c_str());

        // inputs info
        auto subnetworkInputsNode = subnetworkNode.append_child("inputs");
        auto inputInfo = subnet.getInputsInfo();
        for (auto&& input : inputInfo) {
            auto inputNode = subnetworkInputsNode.append_child("input");
            inputNode.append_attribute("name").set_value(input.first.c_str());
            inputNode.append_attribute("precision").set_value(input.second->getPrecision().name());
        }

        // outputs info
        auto subnetworkOutputsNode = subnetworkNode.append_child("outputs");
        auto outputInfo = subnet.getOutputsInfo();
        for (auto&& output : outputInfo) {
            auto outputNode = subnetworkOutputsNode.append_child("output");
            outputNode.append_attribute("name").set_value(output.first.c_str());
            outputNode.append_attribute("precision").set_value(output.second->getPrecision().name());
        }
    }

    auto configsNode = heteroNode.append_child("configs");
    for (auto&& config : _config) {
        auto configNode = configsNode.append_child("config");
        configNode.append_attribute("key").set_value(config.first.c_str());
        configNode.append_attribute("value").set_value(config.second.c_str());
    }

    auto blobNamesNode = heteroNode.append_child("blob_names_map");
    for (auto&& kvp : _blobNameMap) {
        auto blobNameNode = blobNamesNode.append_child("blob_name_map");
        blobNameNode.append_attribute("key").set_value(kvp.first.c_str());
        blobNameNode.append_attribute("value").set_value(kvp.second.c_str());
    }

    doc.save(heteroModel, nullptr, pugi::format_raw);
    doc.reset();
    heteroModel << std::endl;

    for (auto&& subnetwork : _networks) {
        if (_heteroPlugin->GetCore()->DeviceSupportsImportExport(subnetwork._device)) {
            subnetwork._network->Export(heteroModel);
        } else {
            auto subnet = subnetwork._clonedNetwork;
            if (!subnet.getFunction()) {
                IE_THROW() << "Hetero plugin supports only ngraph function representation";
            }

            // Note: custom ngraph extensions are not supported
            std::stringstream xmlFile, binFile;
            ngraph::pass::Serialize serializer(xmlFile, binFile,
                ngraph::pass::Serialize::Version::IR_V10);
            serializer.run_on_function(subnet.getFunction());

            auto m_constants = binFile.str();
            auto m_model = xmlFile.str();

            auto dataSize = static_cast<std::uint64_t>(m_model.size());
            heteroModel.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            heteroModel.write(m_model.c_str(), dataSize);

            dataSize = static_cast<std::uint64_t>(m_constants.size());
            heteroModel.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            heteroModel.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);
        }
    }
}

IInferRequestInternal::Ptr HeteroExecutableNetwork::CreateInferRequestImpl(
        InputsDataMap networkInputs,
        OutputsDataMap networkOutputs) {
    HeteroInferRequest::SubRequestsList inferRequests;
    int index = 0;
    for (auto&& subnetwork : _networks) {
        HeteroInferRequest::SubRequestDesc desc;
        desc._network = subnetwork._network;
        desc._profilingTask = openvino::itt::handle("Infer" + std::to_string(index++));
        inferRequests.push_back(desc);
    }
    return std::make_shared<HeteroInferRequest>(networkInputs,
                                                networkOutputs,
                                                inferRequests,
                                                _blobNameMap);
}

IInferRequestInternal::Ptr HeteroExecutableNetwork::CreateInferRequest() {
    return CreateAsyncInferRequestFromSync<HeteroAsyncInferRequest>();
}

InferenceEngine::Parameter HeteroExecutableNetwork::GetConfig(const std::string &name) const {
    InferenceEngine::Parameter result;
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
        for (auto&& desc : _networks) {
            auto execNetwork = desc._network;
            auto param = execNetwork->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            for (auto && configKey : param.as<std::vector<std::string>>()) {
                if (configKey == name) {
                    return execNetwork->GetConfig(configKey);
                }
            }
        }

        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }

    return result;
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

InferenceEngine::Parameter HeteroExecutableNetwork::GetMetric(const std::string &name) const {
    if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> heteroMetrics = {
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)
        };

        {
            std::vector<::Metrics> pluginMetrics;
            for (auto&& desc : _networks) {
                auto execNetwork = desc._network;
                auto param = execNetwork->GetMetric(METRIC_KEY(SUPPORTED_METRICS));
                ::Metrics metrics;
                for (auto && metricName : param.as<std::vector<std::string>>()) {
                    metrics[metricName] = execNetwork->GetMetric(metricName);
                }
                pluginMetrics.push_back(std::move(metrics));
            }

            collectPluginMetrics(heteroMetrics, pluginMetrics);
        }

        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, heteroMetrics);
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> heteroConfigKeys = {
            "TARGET_FALLBACK",
            HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)
        };

        {
            std::vector<::Metrics> pluginConfigKeys;
            for (auto&& desc : _networks) {
                auto execNetwork = desc._network;
                auto param = execNetwork->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                ::Metrics configKeys;
                for (auto && metricName : param.as<std::vector<std::string>>()) {
                    configKeys[metricName] = execNetwork->GetConfig(metricName);
                }
                pluginConfigKeys.push_back(std::move(configKeys));
            }

            collectPluginMetrics(heteroConfigKeys, pluginConfigKeys);
        }

        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, heteroConfigKeys);
    } else if (EXEC_NETWORK_METRIC_KEY(NETWORK_NAME) == name) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _name);
    } else if (EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        unsigned int value = 0u;
        for (auto&& desc : _networks) {
            value = std::max(value, desc._network->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>());
        }
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        // find metric key among plugin metrics
        for (auto&& desc : _networks) {
            auto execNetwork = desc._network;
            auto param = execNetwork->GetMetric(METRIC_KEY(SUPPORTED_METRICS));
            for (auto && metricKey : param.as<std::vector<std::string>>()) {
                if (metricKey == name) {
                    return execNetwork->GetMetric(metricKey);
                }
            }
        }

        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}
