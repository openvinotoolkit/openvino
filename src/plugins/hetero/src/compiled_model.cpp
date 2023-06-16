// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "infer_request.hpp"
#include "itt.hpp"
#include "plugin.hpp"

// #include "perf_counter.hpp"
#include "converter_utils.hpp"
#include "graph_debug_dump.hpp"
#include "ie_algorithm.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"
#include "xml_parse_utils.h"

template <typename T>
using NodeMap = std::unordered_map<ngraph::Node*, T>;

ov::hetero::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg,
                                         bool loaded_from_cache)
    : ov::ICompiledModel(model, plugin),
      m_cfg(cfg),
      m_model(model),
      m_name(model->get_name()),
      m_loaded_from_cache(loaded_from_cache) {
    try {
        bool dumpDotFile = false;
        if (std::getenv("OPENVINO_HETERO_VISUALIZE")) {
            dumpDotFile = true;
        } else {
            dumpDotFile = m_cfg.dump_graph;
        }

        ov::SupportedOpsMap queryNetworkResult;
        auto orderedOps = m_model->get_ordered_ops();

        bool allEmpty = true;
        // Get user defined affinity
        for (auto&& node : orderedOps) {
            auto& nodeInfo = node->get_rt_info();
            auto itInfo = nodeInfo.find("affinity");
            if (itInfo != nodeInfo.end()) {
                OPENVINO_ASSERT(itInfo->second.is<std::string>());
                queryNetworkResult.emplace(node->get_friendly_name(), itInfo->second.as<std::string>());
                allEmpty = false;
            }
        }

        if (queryNetworkResult.empty())
            queryNetworkResult = plugin->query_model(model, m_cfg.GetHeteroConfig());

        using Input = ov::Input<ov::Node>;
        using NodeSet = std::unordered_set<ov::Node*>;
        using InputSet = std::set<Input>;

        auto InputNode = [](const Input& input) {
            return input.get_source_output().get_node();
        };

        std::unordered_set<std::string> devices;
        NodeMap<std::string> affinities;
        // Check that all nodes has user or plugin defined affinities
        for (auto&& node : orderedOps) {
            auto itAffinity = queryNetworkResult.find(node->get_friendly_name());
            if (itAffinity != queryNetworkResult.end()) {
                affinities[node.get()] = itAffinity->second;
                devices.emplace(itAffinity->second);
            } else if (allEmpty) {
                IE_THROW() << "Hetero device used default fallback policy, but some layers eg: \n(Name:"
                           << node->get_friendly_name() << ", Type: " << node->get_type_name()
                           << ") were not able to be assigned on any pointed device.\n"
                           << "It happened because these layers are not supported in plugins by default.\n"
                           << "You need to implement custom layers to support them.";
            } else {
                IE_THROW() << "Network passed to LoadNetwork has affinity assigned, but some layers eg: \n(Name:"
                           << node->get_friendly_name() << ", Type: " << node->get_type_name()
                           << ") were not assigned to any device.\n"
                           << "It might happen if you assigned layers manually and missed some layers or\n"
                           << "if you used some automatic assigning mode which decided that these layers are not\n"
                           << "supported by any plugin";
            }
        }

        if (dumpDotFile) {
            ov::hetero::debug::dump_affinities(model, queryNetworkResult, devices);
        }

        NodeMap<InputSet> nodeInputDependencies;
        NodeSet graphInputNodes;
        InputSet subgraphInputs;
        // Get all subgraph inputs using just node affinities. Also collect transitive closure
        for (auto&& node : orderedOps) {
            if (ov::op::util::is_parameter(node) || ov::op::util::is_constant(node)) {
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
                    if (!InferenceEngine::details::contains(subgraphInputs,
                                                            input)) {  // TODO vurusovs REPLACE with ov::util::contains
                        inputs.emplace_back(std::move(input));
                    }
                }
                if (inputs.empty()) {
                    subgraphIds.push_back(static_cast<int>(subgraphIds.size()));
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
        for (std::size_t prevSubgraphs = 0, cyclicSplitStep = 0; prevSubgraphs != subgraphInputs.size();
             ++cyclicSplitStep) {
            OPENVINO_ASSERT(cyclicSplitStep < orderedOps.size());
            prevSubgraphs = subgraphInputs.size();
            auto subgraphIds = CollectSubgraphs();
            // All inputs that belong to the same subgraph as node
            std::unordered_map<ov::Node*, InputSet> nodeSubgraphInputDependencies;
            // All inputs that depends on the same subgraph as node
            std::unordered_map<ov::Node*, InputSet> nodeSubgraphCyclicInputDependencies;
            for (auto&& node : orderedOps) {
                auto& nodeSubgraphInputDependency = nodeSubgraphInputDependencies[node.get()];
                auto allNodeSubgraphInputs =
                    InferenceEngine::details::Intersection(nodeInputDependencies[node.get()], subgraphInputs);
                for (auto&& subgraphInput : allNodeSubgraphInputs) {
                    if (subgraphIds[node.get()] == subgraphIds[subgraphInput.get_node()]) {
                        nodeSubgraphInputDependency.emplace(subgraphInput);
                    }
                }
                auto& nodeSubgraphCyclicInputDependency = nodeSubgraphCyclicInputDependencies[node.get()];
                for (auto&& subgraphInput : allNodeSubgraphInputs) {
                    if (!ov::op::util::is_parameter(subgraphInput.get_node()) &&
                        !ov::op::util::is_constant(subgraphInput.get_node()) &&
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
                        auto& inputNodeSubgraphCyclicInputDependency =
                            nodeSubgraphCyclicInputDependencies[InputNode(input)];
                        auto& inputNodeSubgraphInputDependency = nodeSubgraphInputDependencies[InputNode(input)];
                        if (!InferenceEngine::details::Intersects(nodeSubgraphCyclicInputDependency,
                                                                  inputNodeSubgraphCyclicInputDependency) &&
                            InferenceEngine::details::Intersects(cyclicInputsDependencies,
                                                                 inputNodeSubgraphInputDependency)) {
                            subgraphInputs.insert(input);
                        }
                    }
                }
            }
        }

        auto subgraphIds = CollectSubgraphs();

        if (dumpDotFile) {
            std::map<std::string, int> map_id;
            for (auto&& v : subgraphIds) {
                map_id.emplace(v.first->get_friendly_name(), v.second);
            }
            ov::hetero::debug::dump_subgraphs(model, queryNetworkResult, map_id);
        }

        // Break graph using insertion of result parameter split
        NodeMap<ngraph::Node*> subgraphParameterToPrevResult;
        std::vector<std::shared_ptr<ngraph::op::Result>> results;
        {
            std::set<ngraph::Output<ngraph::Node>> subgraphOutputs;
            for (auto&& input : subgraphInputs) {
                if (!ov::op::util::is_parameter(input.get_node()) && !ov::op::util::is_constant(input.get_node())) {
                    subgraphOutputs.insert(input.get_source_output());
                }
            }
            for (auto&& output : subgraphOutputs) {
                auto output_subgraph_id = subgraphIds.at(output.get_node());
                auto inputs = output.get_target_inputs();
                // Collect input subsets from other subgraphs. Each subset of inputs belongs to the same subgraph
                std::map<int, std::set<ngraph::Input<ngraph::Node>>> input_subsets;
                for (auto&& input : inputs) {
                    auto input_subgraph_id = subgraphIds.at(input.get_node());
                    if (output_subgraph_id != input_subgraph_id) {
                        input_subsets[input_subgraph_id].emplace(input);
                    }
                }
                // for each subset of inputs create separate Result operation if subset belongs to other
                for (auto&& input_subset : input_subsets) {
                    auto result = std::make_shared<ngraph::op::Result>(output);
                    result->set_friendly_name(output.get_node()->get_friendly_name() + "_" +
                                              std::to_string(output.get_index()) + "_" +
                                              std::to_string(input_subset.first) + "_result");
                    ov::copy_runtime_info(output.get_node_shared_ptr(), result);
                    subgraphIds.emplace(result.get(), output_subgraph_id);
                    results.push_back(result);
                    for (auto&& input : input_subset.second) {
                        output.remove_target_input(input);
                        auto parameter = std::make_shared<ngraph::op::Parameter>(output.get_element_type(),
                                                                                 output.get_partial_shape());
                        parameter->set_friendly_name(input.get_node()->get_friendly_name() + "_" +
                                                     std::to_string(input.get_index()) + "_parameter");
                        ov::copy_runtime_info(input.get_node()->shared_from_this(), parameter);
                        input.replace_source_output(parameter->output(0));
                        subgraphIds.emplace(parameter.get(), input_subset.first);
                        subgraphParameterToPrevResult.emplace(parameter.get(), result.get());
                        _mapOutputToInput.emplace(  // TODO: reuse subgraphParameterToPrevResult
                            result,
                            parameter
                        );
                        _blobNameMap.emplace(
                            parameter->get_friendly_name(),
                            output.get_node()->get_friendly_name() + ((output.get_node()->get_output_size() != 1)
                                                                          ? ("." + std::to_string(output.get_index()))
                                                                          : std::string{}));
                    }
                }
            }
        }

        struct Subgraph {
            ngraph::ResultVector _results;
            ngraph::ParameterVector _parameters;
            ngraph::SinkVector _sinks;
            std::string _affinity;
        };
        std::unordered_map<int, Subgraph> subgraphs;
        // Extracts subgraph parameters, results and affinities
        for (auto&& subgraphIdPtrValue : subgraphIds) {
            auto node = subgraphIdPtrValue.first;
            auto& subgraph = subgraphs[subgraphIdPtrValue.second];
            if (ov::op::util::is_output(node)) {
                subgraph._results.emplace_back(
                    std::dynamic_pointer_cast<ngraph::op::v0::Result>(node->shared_from_this()));
            } else if (ov::op::util::is_parameter(node)) {
                subgraph._parameters.emplace_back(
                    std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(node->shared_from_this()));
            } else if (ov::op::util::is_sink(node)) {
                subgraph._sinks.emplace_back(std::dynamic_pointer_cast<ngraph::op::Sink>(node->shared_from_this()));
            }
            auto itAffinity = affinities.find(node);
            if (itAffinity != affinities.end()) {
                subgraph._affinity = itAffinity->second;
            }
        }
        results = {};

        // Subgraph topological sort
        std::vector<Subgraph> allSubgraphs;
        for (auto&& subgraph : subgraphs) {
            allSubgraphs.emplace_back(std::move(subgraph.second));
        }

        std::vector<Subgraph> orderedSubgraphs;
        NodeSet prevResults;
        size_t subgraphTopoSortsStep = 0;
        do {
            OPENVINO_ASSERT(subgraphTopoSortsStep < subgraphs.size());
            ++subgraphTopoSortsStep;
            std::vector<Subgraph> newOrderedSubgraphs;
            auto IsOrderedSubGraph = [&](const Subgraph& subgraph) {
                auto& parameters = subgraph._parameters;
                return std::all_of(parameters.begin(),
                                   parameters.end(),
                                   [&](const ngraph::ParameterVector::value_type& parameter) {
                                       return InferenceEngine::details::contains(graphInputNodes, parameter.get()) ||
                                              InferenceEngine::details::contains(
                                                  prevResults,
                                                  subgraphParameterToPrevResult[parameter.get()]);
                                   });
            };
            std::remove_copy_if(std::begin(allSubgraphs),
                                std::end(allSubgraphs),
                                std::back_inserter(newOrderedSubgraphs),
                                [&](const Subgraph& subgraph) {
                                    return !IsOrderedSubGraph(subgraph);
                                });
            allSubgraphs.erase(std::remove_if(std::begin(allSubgraphs), std::end(allSubgraphs), IsOrderedSubGraph),
                               std::end(allSubgraphs));
            for (auto&& subgraph : newOrderedSubgraphs) {
                for (auto&& result : subgraph._results) {
                    prevResults.insert(result.get());
                }
            }
            std::move(std::begin(newOrderedSubgraphs),
                      std::end(newOrderedSubgraphs),
                      std::back_inserter(orderedSubgraphs));
        } while (!allSubgraphs.empty());

        // Prepare mapping between original inputs/outputs and compiled
        // submodels inputs/outputs. Example:
        // original input 0 -> submodel 0 input 0,
        // original input 1 -> submodel 1 input 0,
        // original output 0 -> submodel 1 output 0.
        //
        // Mapping is required only because before compilation
        // submodel may be preprocessed (if legacy API used),
        // so original inputs/outputs != submodels inputs/outputs
        m_inputs_to_submodel_inputs.resize(m_inputs.size());
        m_outputs_to_submodel_outputs.resize(m_outputs.size());
        for (size_t id = 0; id < orderedSubgraphs.size(); id++) {
            for (size_t i = 0; i < orderedSubgraphs[id]._parameters.size(); i++) {
                const auto& input_node = std::dynamic_pointer_cast<ov::Node>(orderedSubgraphs[id]._parameters[i]);
                for (size_t j = 0; j < m_inputs.size(); j++)
                    if (input_node == m_inputs[j].get_node_shared_ptr())
                        m_inputs_to_submodel_inputs[j] = {id, i};
            }
            for (size_t i = 0; i < orderedSubgraphs[id]._results.size(); i++) {
                const auto& input_node = std::dynamic_pointer_cast<ov::Node>(orderedSubgraphs[id]._results[i]);
                for (size_t j = 0; j < m_outputs.size(); j++)
                    if (input_node == m_outputs[j].get_node_shared_ptr())
                        m_outputs_to_submodel_outputs[j] = {id, i};
            }
        }

        // Prepare mapping between manually splitted inputs/outputs
        // to connect tensors between compiled submodels
        for (const auto& kvp: _mapOutputToInput) {
            const auto& intermed_output = kvp.first;
            const auto& intermed_input = kvp.second;
            for (size_t id = 0; id < orderedSubgraphs.size(); id++) {
                const auto& out_it = std::find(orderedSubgraphs[id]._results.begin(), orderedSubgraphs[id]._results.end(), intermed_output);
                if (out_it != orderedSubgraphs[id]._results.end()) {
                    for (size_t id2 = 0; id2 < orderedSubgraphs.size(); id2++) {
                        if (id2 == id)
                            continue;
                        const auto& in_it = std::find(orderedSubgraphs[id2]._parameters.begin(), orderedSubgraphs[id2]._parameters.end(), intermed_input);
                        if (in_it != orderedSubgraphs[id2]._parameters.end()) {
                            auto out_idx = std::distance(orderedSubgraphs[id]._results.begin(), out_it);
                            auto in_idx = std::distance(orderedSubgraphs[id2]._parameters.begin(), in_it);
                            m_submodels_output_to_input[{id, out_idx}] = {id2, in_idx};
                        }
                    }
                }
            }
        }

        ov::ParameterVector externalInputsData = model->get_parameters();
        ov::ResultVector externalOutputsData = model->get_results();

        m_networks.resize(orderedSubgraphs.size());
        std::vector<std::shared_ptr<ov::Model>> subFunctions(orderedSubgraphs.size());
        int id = 0;
        for (auto&& subgraph : orderedSubgraphs) {
            m_networks[id]._device = subgraph._affinity;
            subFunctions[id] = std::make_shared<ov::Model>(subgraph._results,
                                                           subgraph._sinks,
                                                           subgraph._parameters,
                                                           m_name + '_' + std::to_string(id));
            m_networks[id]._clonedNetwork = subFunctions[id];

            // update of pre-processing info
            // auto clonedInputs = _networks[id]._clonedNetwork.getInputsInfo();
            // for (auto&& externalInput : externalInputsData) {
            //     auto itClonedInput = clonedInputs.find(externalInput.first);
            //     if (itClonedInput != clonedInputs.end() && nullptr != itClonedInput->second) {
            //         itClonedInput->second->getPreProcess() = externalInput.second->getPreProcess();
            //         itClonedInput->second->setPrecision(externalInput.second->getPrecision());
            //         itClonedInput->second->setLayout(externalInput.second->getLayout());
            //     }
            // }
            // // update output info
            // auto clonedOutputs = _networks[id]._clonedNetwork.getOutputsInfo();
            // for (auto&& externalOutput : externalOutputsData) {
            //     auto itClonedOutput = clonedOutputs.find(externalOutput.first);
            //     if (itClonedOutput != clonedOutputs.end() && nullptr != itClonedOutput->second) {
            //         itClonedOutput->second->setPrecision(externalOutput.second->getPrecision());
            //         itClonedOutput->second->setLayout(externalOutput.second->getLayout());
            //     }
            // }

            // auto toLegacyType = [](const ngraph::element::Type& ngraph_type) {
            //     return (ngraph_type == ngraph::element::f16 || ngraph_type == ngraph::element::bf16) ?
            //     ngraph::element::f32
            //                                                                                          : ngraph_type;
            // };

            // CNNNetwork converts input and output types to preserve legacy behaviour
            // Here io types are reverted to ngraph types with some common plugin behaviour assumption
            // defined in `toLegacyType()`
            // for (auto&& input : clonedInputs) {
            //     if (!InferenceEngine::details::contains(externalInputsData, input.first)) {
            //         for (auto&& parameter : subgraph._parameters) {
            //             auto name = parameter->get_friendly_name();
            //             if (parameter->get_friendly_name() == input.first) {
            //                 input.second->setPrecision(
            //                     InferenceEngine::details::convertPrecision(toLegacyType(parameter->get_element_type())));
            //             }
            //         }
            //     }
            // }
            // for (auto&& output : clonedOutputs) {
            //     if (!InferenceEngine::details::contains(externalOutputsData, output.first)) {
            //         for (auto&& result : subgraph._results) {
            //             auto source_output = result->input_value(0);
            //             auto output_name = ov::op::util::create_ie_output_name(source_output);
            //             if (output_name == output.first) {
            //                 output.second->setPrecision(
            //                     InferenceEngine::details::convertPrecision(toLegacyType(source_output.get_element_type())));
            //             }
            //         }
            //     }
            // }
            ++id;
        }
        for (auto&& network : m_networks) {
            auto metaDevices = get_hetero_plugin()->get_properties_per_device(network._device, m_cfg.GetDeviceConfig());

            // disable caching for subgraphs, because the whole HETERO model is cached
            auto device_config = metaDevices[network._device];
            device_config[ov::cache_dir.name()] = "";

            network._network =
                plugin->get_core()->compile_model(network._clonedNetwork, network._device, device_config);
        }

    } catch (const InferenceEngine::Exception& e) {
        // Some transformations can throw legacy exception
        OPENVINO_THROW(e.what());
    } catch (const std::exception& e) {
        OPENVINO_THROW("Standard exception from compilation library: ", e.what());
    } catch (...) {
        OPENVINO_THROW("Generic exception is thrown");
    }
}

ov::hetero::CompiledModel::CompiledModel(std::istream& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg,
                                         bool loaded_from_cache)
    : ov::ICompiledModel(nullptr, plugin),
      m_cfg(cfg),
      m_model(nullptr),
      m_name(),
      m_loaded_from_cache(loaded_from_cache) {
    std::string heteroXmlStr;
    std::getline(model, heteroXmlStr);

    pugi::xml_document heteroXmlDoc;
    pugi::xml_parse_result res = heteroXmlDoc.load_string(heteroXmlStr.c_str());

    if (res.status != pugi::status_ok) {
        IE_THROW(NetworkNotRead) << "Error reading HETERO device xml header";
    }

    using namespace pugixml::utils;

    pugi::xml_node heteroNode = heteroXmlDoc.document_element();
    m_name = GetStrAttr(heteroNode, "name");

    ov::AnyMap properties;
    auto heteroConfigsNode = heteroNode.child("hetero_config");
    FOREACH_CHILD (heteroConfigNode, heteroConfigsNode, "config") {
        properties.emplace(GetStrAttr(heteroConfigNode, "key"), GetStrAttr(heteroConfigNode, "value"));
    }

    auto deviceConfigsNode = heteroNode.child("device_config");
    FOREACH_CHILD (deviceConfigNode, deviceConfigsNode, "config") {
        properties.emplace(GetStrAttr(deviceConfigNode, "key"), GetStrAttr(deviceConfigNode, "value"));
    }

    // Erase all "hetero" properties from `properties`
    // to fill `m_cfg` and leave only properties for
    // underlying devices
    m_cfg = ov::hetero::Configuration(properties, m_cfg);

    auto blobNamesNode = heteroNode.child("blob_names_map");
    FOREACH_CHILD (blobNameNode, blobNamesNode, "blob_name_map") {
        _blobNameMap.emplace(GetStrAttr(blobNameNode, "key"), GetStrAttr(blobNameNode, "value"));
    }

    pugi::xml_node subnetworksNode = heteroNode.child("subnetworks");
    FOREACH_CHILD (subnetworkNode, subnetworksNode, "subnetwork") {
        auto deviceName = GetStrAttr(subnetworkNode, "device");

        auto metaDevices = get_hetero_plugin()->get_properties_per_device(deviceName, properties);
        assert(metaDevices.size() == 1);
        auto& loadConfig = metaDevices[deviceName];

        ov::SoPtr<ov::ICompiledModel> compiled_model;
        std::shared_ptr<ov::Model> ov_model;

        bool loaded = false;
        if (get_hetero_plugin()->device_supports_model_caching(deviceName)) {
            compiled_model = plugin->get_core()->import_model(model, deviceName, loadConfig);
        } else {
            // read XML content
            std::string xmlString;
            std::uint64_t dataSize = 0;
            model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            xmlString.resize(dataSize);
            model.read(const_cast<char*>(xmlString.c_str()), dataSize);

            /// read blob content
            ov::Tensor weights;
            model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            if (0 != dataSize) {
                weights = ov::Tensor(ov::element::from<char>(), ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
                model.read(weights.data<char>(), dataSize);
            }

            auto ov_model = plugin->get_core()->read_model(xmlString, weights);
            compiled_model = plugin->get_core()->compile_model(ov_model, deviceName, loadConfig);
            loaded = true;
        }

        m_networks.emplace_back(ov::hetero::CompiledModel::NetworkDesc{
            deviceName,
            loaded ? ov_model : std::shared_ptr<ov::Model>{},
            compiled_model,
        });
    }
    const auto parseNode = [](const pugi::xml_node& xml_node, bool is_param) -> std::shared_ptr<const ov::Node> {
        const std::string operation_name = GetStrAttr(xml_node, "operation_name");
        const auto elementType = ov::EnumNames<ov::element::Type_t>::as_enum(GetStrAttr(xml_node, "element_type"));

        std::vector<ov::Dimension> partialShape;
        pugi::xml_node partialShapeNode = xml_node.child("partial_shape");
        FOREACH_CHILD (dimNode, partialShapeNode, "dim") {
            partialShape.emplace_back(ov::Dimension(GetInt64Attr(dimNode, "value")));
        }

        pugi::xml_node tensorNamesNode = xml_node.child("tensor_names");
        std::unordered_set<std::string> tensorNames;
        FOREACH_CHILD (tensorNameNode, tensorNamesNode, "tensor_name") {
            tensorNames.insert(GetStrAttr(tensorNameNode, "value"));
        }

        std::shared_ptr<ov::Node> node = std::make_shared<ov::op::v0::Parameter>(elementType, partialShape);
        // For result operation_name is name of previous operation
        node->set_friendly_name(operation_name);
        if (!is_param)
            node = std::make_shared<ov::op::v0::Result>(node);
        node->output(0).get_tensor().add_names(tensorNames);

        return node;
    };
    (void)parseNode;

    pugi::xml_node parametersNode = heteroNode.child("parameters");
    FOREACH_CHILD (parameterNode, parametersNode, "parameter") {
        m_inputs.emplace_back(parseNode(parameterNode, true));
    }

    pugi::xml_node resultsNode = heteroNode.child("results");
    FOREACH_CHILD (resultNode, resultsNode, "result") { m_outputs.emplace_back(parseNode(resultNode, false)); }
}

std::shared_ptr<ov::ISyncInferRequest> ov::hetero::CompiledModel::create_sync_infer_request() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IAsyncInferRequest> ov::hetero::CompiledModel::create_infer_request() const {
    auto convert = [](const std::vector<ov::Output<const ov::Node>>& vec) {
        std::vector<std::shared_ptr<const ov::Node>> new_vec;
        for (const auto& node : vec) {
            new_vec.emplace_back(node.get_node_shared_ptr());
        }
        return new_vec;
    };

    const std::vector<std::shared_ptr<const ov::Node>> net_inputs = convert(inputs());
    const std::vector<std::shared_ptr<const ov::Node>> net_outputs = convert(outputs());

    HeteroPlugin::HeteroInferRequest::SubRequestsList inferRequests;
    int index = 0;
    for (auto&& subnetwork : m_networks) {
        HeteroPlugin::HeteroInferRequest::SubRequestDesc desc;
        auto legacy_exe_net = ov::legacy_convert::convert_compiled_model(subnetwork._network._ptr);
        desc._network = {legacy_exe_net, subnetwork._network._so};
        desc._profilingTask = openvino::itt::handle("Infer" + std::to_string(index++));
        inferRequests.push_back(desc);
    }
    auto legacy_sync_infer_req =
        std::make_shared<HeteroPlugin::HeteroInferRequest>(net_inputs, net_outputs, inferRequests, _blobNameMap);
    auto exe_ptr =
        ov::legacy_convert::convert_compiled_model(std::const_pointer_cast<ov::ICompiledModel>(shared_from_this()));
    legacy_sync_infer_req->setPointerToExecutableNetworkInternal(exe_ptr);
    auto legacy_async_infer_req = std::make_shared<HeteroPlugin::HeteroAsyncInferRequest>(
        legacy_sync_infer_req,
        std::dynamic_pointer_cast<InferenceEngine::ITaskExecutor>(get_task_executor()),
        std::dynamic_pointer_cast<InferenceEngine::ITaskExecutor>(get_callback_executor()));
    auto req = ov::legacy_convert::convert_infer_request(legacy_async_infer_req);
    return req;
}

void ov::hetero::CompiledModel::set_property(const ov::AnyMap& properties) {
    auto temp_properties(properties);
    m_cfg = Configuration{temp_properties, m_cfg};
}

std::shared_ptr<const ov::Model> ov::hetero::CompiledModel::get_runtime_model() const {
    auto model = m_model->clone();
    // Add execution information into the model
    size_t exec_order = 0;
    for (const auto& op : model->get_ordered_ops()) {
        auto& info = op->get_rt_info();
        // const auto& it = info.find(ov::runtime::interpreter::PERF_COUNTER_NAME);
        // OPENVINO_ASSERT(it != info.end(), "Operation ", op, " doesn't contain performance counter");
        // auto perf_count = it->second.as<std::shared_ptr<ov::runtime::interpreter::PerfCounter>>();
        // OPENVINO_ASSERT(perf_count, "Performance counter is empty");
        info[ov::exec_model_info::LAYER_TYPE] = op->get_type_info().name;
        info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(exec_order++);
        info[ov::exec_model_info::IMPL_TYPE] = "ref";
        // TODO vurusovs NEED TO ENABLE???
        // info[ov::exec_model_info::PERF_COUNTER] = m_cfg.perf_count && perf_count && perf_count->avg() != 0
        //                                               ? std::to_string(perf_count->avg())
        //                                               : "not_executed";

        std::string original_names = ov::getFusedNames(op);
        if (original_names.empty()) {
            original_names = op->get_friendly_name();
        } else if (original_names.find(op->get_friendly_name()) == std::string::npos) {
            original_names = op->get_friendly_name() + "," + original_names;
        }
        info[ov::exec_model_info::ORIGINAL_NAMES] = original_names;
    }
    return model;
}

std::shared_ptr<const ov::hetero::Plugin> ov::hetero::CompiledModel::get_hetero_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto hetero_plugin = std::static_pointer_cast<const ov::hetero::Plugin>(plugin);
    OPENVINO_ASSERT(hetero_plugin);
    return hetero_plugin;
}

ov::Any ov::hetero::CompiledModel::get_property(const std::string& name) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache};
        return ro_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    
    if (ov::supported_properties == name) {
        auto supported_properties = default_ro_properties();
        add_ro_properties(ov::supported_properties.name(), supported_properties);
        add_ro_properties(ov::device::properties.name(), supported_properties);
        add_ro_properties(ov::device::priorities.name(), supported_properties);
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();
        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        return to_string_vector(metrics);
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        return to_string_vector(m_cfg.GetSupported());
    } else if (ov::device::properties == name) {
        ov::AnyMap all_devices = {};
        for (auto&& subnetwork : m_networks) {
            ov::AnyMap device_properties = {};
            if (all_devices.count(subnetwork._device) == 0) {
                auto device_supported_metrics = subnetwork._network->get_property(METRIC_KEY(SUPPORTED_METRICS));
                for (auto&& property_name : device_supported_metrics.as<std::vector<std::string>>()) {
                    device_properties[property_name] = subnetwork._network->get_property(property_name);
                }
                auto device_supported_configs = subnetwork._network->get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                for (auto&& property_name : device_supported_configs.as<std::vector<std::string>>()) {
                    device_properties[property_name] = subnetwork._network->get_property(property_name);
                }
                all_devices[subnetwork._device] = device_properties;
            }
        }
        return all_devices;
    } else if (ov::model_name == name) {
        return decltype(ov::model_name)::value_type(m_name);
    } else if (ov::loaded_from_cache == name) {
        return decltype(ov::loaded_from_cache)::value_type{m_loaded_from_cache};
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = 0u;
        for (auto&& desc : m_networks) {
            value = std::max(value,
                             desc._network->get_property(ov::optimal_number_of_infer_requests.name()).as<unsigned int>());
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type{value};
    } else if (ov::execution_devices == name) {
        std::vector<std::string> device_names;
        std::set<std::string> s;
        for (auto&& subnetwork : m_networks) {
            if (s.count(subnetwork._device) != 0)
                continue;
            s.insert(subnetwork._device);
            device_names.push_back(subnetwork._device);
        }
        return decltype(ov::execution_devices)::value_type{device_names};
    }
    return m_cfg.Get(name);
}

void ov::hetero::CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "CompiledModel::export_model");

    pugi::xml_document doc;
    auto heteroNode = doc.append_child("hetero");
    heteroNode.append_attribute("name").set_value(m_name.c_str());

    const auto serializeNode = [&](const std::shared_ptr<const ov::Node>& node, pugi::xml_node& xml_node) {
        const bool is_result = ov::is_type<ov::op::v0::Result>(node);
        const std::string name =
            is_result ? ov::op::util::create_ie_output_name(node->input_value(0)) : node->get_friendly_name();
        xml_node.append_attribute("operation_name").set_value(name.c_str());
        xml_node.append_attribute("element_type").set_value(node->get_output_element_type(0).get_type_name().c_str());

        const auto& pShape = node->get_output_partial_shape(0);
        OPENVINO_ASSERT(pShape.rank().is_static(), "Serialization of shapes with dynamic rank is not supported");
        auto partialShapeNode = xml_node.append_child("partial_shape");
        for (auto&& dim : node->get_output_partial_shape(0)) {
            if (dim.is_dynamic())
                partialShapeNode.append_child("dim").append_attribute("value").set_value("-1");
            else
                partialShapeNode.append_child("dim").append_attribute("value").set_value(
                    std::to_string(dim.get_length()).c_str());
        }

        auto tensorNamesNode = xml_node.append_child("tensor_names");
        for (auto& tensorName : node->get_output_tensor(0).get_names()) {
            tensorNamesNode.append_child("tensor_name").append_attribute("value").set_value(tensorName.c_str());
        }
    };

    // ngraph parameters info
    auto subnetworkInputs = heteroNode.append_child("parameters");
    for (auto&& input : inputs()) {
        auto parameterNode = subnetworkInputs.append_child("parameter");
        serializeNode(input.get_node_shared_ptr(), parameterNode);
    }

    // ngraph results info
    auto subnetworkResultsNode = heteroNode.append_child("results");
    for (auto&& output : outputs()) {
        auto resultNode = subnetworkResultsNode.append_child("result");
        serializeNode(output.get_node_shared_ptr(), resultNode);
    }

    auto subnetworksNode = heteroNode.append_child("subnetworks");
    for (auto&& subnetwork : m_networks) {
        auto subnet = subnetwork._clonedNetwork;
        OPENVINO_ASSERT(subnet);

        auto subnetworkNode = subnetworksNode.append_child("subnetwork");
        subnetworkNode.append_attribute("device").set_value(subnetwork._device.c_str());
    }

    auto heteroConfigsNode = heteroNode.append_child("hetero_config");
    for (auto&& config : m_cfg.GetHeteroConfig()) {
        auto heteroConfigNode = heteroConfigsNode.append_child("config");
        heteroConfigNode.append_attribute("key").set_value(config.first.c_str());
        heteroConfigNode.append_attribute("value").set_value(config.second.as<std::string>().c_str());
    }

    auto deviceConfigsNode = heteroNode.append_child("device_config");
    for (auto&& config : m_cfg.GetDeviceConfig()) {
        auto deviceConfigNode = deviceConfigsNode.append_child("config");
        deviceConfigNode.append_attribute("key").set_value(config.first.c_str());
        deviceConfigNode.append_attribute("value").set_value(config.second.as<std::string>().c_str());
    }

    auto blobNamesNode = heteroNode.append_child("blob_names_map");
    for (auto&& kvp : _blobNameMap) {
        auto blobNameNode = blobNamesNode.append_child("blob_name_map");
        blobNameNode.append_attribute("key").set_value(kvp.first.c_str());
        blobNameNode.append_attribute("value").set_value(kvp.second.c_str());
    }

    doc.save(model_stream, nullptr, pugi::format_raw);
    doc.reset();
    model_stream << std::endl;

    for (auto&& subnetwork : m_networks) {
        if (get_hetero_plugin()->device_supports_model_caching(subnetwork._device)) {
            subnetwork._network->export_model(model_stream);
        } else {
            auto subnet = subnetwork._clonedNetwork;
            if (!subnet) {
                IE_THROW() << "Hetero device supports only ngraph function representation";
            }

            // Note: custom ngraph extensions are not supported
            std::stringstream xmlFile, binFile;
            // ov::pass::Serialize serializer(xmlFile, binFile, ov::pass::Serialize::Version::IR_V10);
            ov::pass::Serialize serializer(xmlFile, binFile);
            serializer.run_on_model(subnet);

            auto constants = binFile.str();
            auto model = xmlFile.str();

            auto dataSize = static_cast<std::uint64_t>(model.size());
            model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            model_stream.write(model.c_str(), dataSize);

            dataSize = static_cast<std::uint64_t>(constants.size());
            model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            model_stream.write(reinterpret_cast<char*>(&constants[0]), dataSize);
        }
    }
}