// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "graph_debug_dump.hpp"
#include "ie_algorithm.hpp"
#include "ie_plugin_config.hpp"
#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"
#include "xml_parse_utils.h"

template <typename T>
using NodeMap = std::unordered_map<std::shared_ptr<ov::Node>, T>;

ov::hetero::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg,
                                         bool loaded_from_cache)
    : ov::ICompiledModel(model, plugin),
      m_cfg(cfg),
      m_model(model),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(loaded_from_cache) {
    try {
        bool dumpDotFile = m_cfg.dump_graph;
        if (std::getenv("OPENVINO_HETERO_VISUALIZE"))
            dumpDotFile = true;

        ov::SupportedOpsMap queryNetworkResult;
        auto orderedOps = model->get_ordered_ops();

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
            queryNetworkResult = plugin->query_model(model, m_cfg.GetHeteroProperties());

        using Input = ov::Input<ov::Node>;
        using NodeSet = std::unordered_set<std::shared_ptr<ov::Node>>;
        using InputSet = std::set<Input>;

        auto InputNode = [](const Input& input) {
            return input.get_source_output().get_node_shared_ptr();
        };

        std::unordered_set<std::string> devices;
        NodeMap<std::string> affinities;
        // Check that all nodes has user or plugin defined affinities
        for (auto&& node : orderedOps) {
            auto itAffinity = queryNetworkResult.find(node->get_friendly_name());
            if (itAffinity != queryNetworkResult.end()) {
                affinities[node] = itAffinity->second;
                devices.emplace(itAffinity->second);
            } else if (allEmpty) {
                OPENVINO_THROW("Hetero device used default fallback policy, but some layers eg: \n(Name:",
                               node->get_friendly_name(),
                               ", Type: ",
                               node->get_type_name(),
                               ") were not able to be assigned on any pointed device.\n",
                               "It happened because these layers are not supported in plugins by default.\n",
                               "You need to implement custom layers to support them.");
            } else {
                OPENVINO_THROW("Network passed to CompiledModel has affinity assigned, but some layers eg: \n(Name:",
                               node->get_friendly_name(),
                               ", Type: ",
                               node->get_type_name(),
                               ") were not assigned to any device.\n",
                               "It might happen if you assigned layers manually and missed some layers or\n",
                               "if you used some automatic assigning mode which decided that these layers are not\n",
                               "supported by any plugin");
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
                graphInputNodes.insert(node);
                subgraphInputs.insert(Input{node.get(), 0});
                nodeInputDependencies[node].insert(Input{node.get(), 0});
            } else {
                auto inputs = node->inputs();
                auto& nodeInputDependency = nodeInputDependencies[node];
                for (auto&& input : inputs) {
                    nodeInputDependency.insert(input);
                    auto& inputDependency = nodeInputDependencies[InputNode(input)];
                    nodeInputDependency.insert(inputDependency.begin(), inputDependency.end());
                    if (affinities[node] != affinities[InputNode(input)]) {
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
                    subgraphIdPtrs.emplace(node, &(subgraphIds.back()));
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
                    subgraphIdPtrs.emplace(node, firstInputSubgraphIdPtr);
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
            std::unordered_map<std::shared_ptr<ov::Node>, InputSet> nodeSubgraphInputDependencies;
            // All inputs that depends on the same subgraph as node
            std::unordered_map<std::shared_ptr<ov::Node>, InputSet> nodeSubgraphCyclicInputDependencies;
            for (auto&& node : orderedOps) {
                auto& nodeSubgraphInputDependency = nodeSubgraphInputDependencies[node];
                auto allNodeSubgraphInputs =
                    InferenceEngine::details::Intersection(nodeInputDependencies[node], subgraphInputs);
                for (auto&& subgraphInput : allNodeSubgraphInputs) {
                    if (subgraphIds[node] == subgraphIds[subgraphInput.get_node()->shared_from_this()]) {
                        nodeSubgraphInputDependency.emplace(subgraphInput);
                    }
                }
                auto& nodeSubgraphCyclicInputDependency = nodeSubgraphCyclicInputDependencies[node];
                for (auto&& subgraphInput : allNodeSubgraphInputs) {
                    if (!ov::op::util::is_parameter(subgraphInput.get_node()) &&
                        !ov::op::util::is_constant(subgraphInput.get_node()) &&
                        subgraphIds[node] == subgraphIds[InputNode(subgraphInput)]) {
                        nodeSubgraphCyclicInputDependency.emplace(subgraphInput);
                    }
                }
            }

            for (auto&& node : orderedOps) {
                auto& nodeSubgraphCyclicInputDependency = nodeSubgraphCyclicInputDependencies[node];
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
        NodeMap<std::shared_ptr<ov::Node>> subgraphParameterToPrevResult;
        std::vector<std::shared_ptr<ov::op::v0::Result>> results;
        {
            std::set<ov::Output<ov::Node>> subgraphOutputs;
            for (auto&& input : subgraphInputs) {
                if (!ov::op::util::is_parameter(input.get_node()) && !ov::op::util::is_constant(input.get_node())) {
                    subgraphOutputs.insert(input.get_source_output());
                }
            }
            for (auto&& output : subgraphOutputs) {
                auto output_subgraph_id = subgraphIds.at(output.get_node_shared_ptr());
                auto inputs = output.get_target_inputs();
                // Collect input subsets from other subgraphs. Each subset of inputs belongs to the same subgraph
                std::map<int, std::set<ov::Input<ov::Node>>> input_subsets;
                for (auto&& input : inputs) {
                    auto input_subgraph_id = subgraphIds.at(input.get_node()->shared_from_this());
                    if (output_subgraph_id != input_subgraph_id) {
                        input_subsets[input_subgraph_id].emplace(input);
                    }
                }
                // for each subset of inputs create separate Result operation if subset belongs to other
                for (auto&& input_subset : input_subsets) {
                    auto result = std::make_shared<ov::op::v0::Result>(output);
                    result->set_friendly_name(output.get_node()->get_friendly_name() + "_" +
                                              std::to_string(output.get_index()) + "_" +
                                              std::to_string(input_subset.first) + "_result");
                    ov::copy_runtime_info(output.get_node_shared_ptr(), result);
                    subgraphIds.emplace(result, output_subgraph_id);
                    results.push_back(result);
                    for (auto&& input : input_subset.second) {
                        output.remove_target_input(input);
                        auto parameter = std::make_shared<ov::op::v0::Parameter>(output.get_element_type(),
                                                                                 output.get_partial_shape());
                        parameter->set_friendly_name(input.get_node()->get_friendly_name() + "_" +
                                                     std::to_string(input.get_index()) + "_parameter");
                        ov::copy_runtime_info(input.get_node()->shared_from_this(), parameter);
                        input.replace_source_output(parameter->output(0));
                        subgraphIds.emplace(parameter, input_subset.first);
                        subgraphParameterToPrevResult.emplace(parameter, result);
                    }
                }
            }
        }

        struct Subgraph {
            ov::ResultVector _results;
            ov::ParameterVector _parameters;
            ov::SinkVector _sinks;
            std::string _affinity;
        };
        std::unordered_map<int, Subgraph> subgraphs;
        // Extracts subgraph parameters, results and affinities
        for (auto&& subgraphIdPtrValue : subgraphIds) {
            auto node = subgraphIdPtrValue.first;
            auto& subgraph = subgraphs[subgraphIdPtrValue.second];
            if (ov::op::util::is_output(node)) {
                subgraph._results.emplace_back(std::dynamic_pointer_cast<ov::op::v0::Result>(node->shared_from_this()));
            } else if (ov::op::util::is_parameter(node)) {
                subgraph._parameters.emplace_back(
                    std::dynamic_pointer_cast<ov::op::v0::Parameter>(node->shared_from_this()));
            } else if (ov::op::util::is_sink(node)) {
                subgraph._sinks.emplace_back(std::dynamic_pointer_cast<ov::op::Sink>(node->shared_from_this()));
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
                                   [&](const ov::ParameterVector::value_type& parameter) {
                                       return InferenceEngine::details::contains(graphInputNodes, parameter) ||
                                              InferenceEngine::details::contains(
                                                  prevResults,
                                                  subgraphParameterToPrevResult[parameter]);
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
                    prevResults.insert(result);
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
        const auto& orig_parameters = model->get_parameters();
        const auto& orig_results = model->get_results();
        m_inputs_to_submodels_inputs.resize(orig_parameters.size());
        m_outputs_to_submodels_outputs.resize(orig_results.size());
        for (size_t id = 0; id < orderedSubgraphs.size(); id++) {
            for (size_t i = 0; i < orderedSubgraphs[id]._parameters.size(); i++) {
                for (size_t j = 0; j < orig_parameters.size(); j++)
                    if (orderedSubgraphs[id]._parameters[i] == orig_parameters[j])
                        m_inputs_to_submodels_inputs[j] = {id, i};
            }
            for (size_t i = 0; i < orderedSubgraphs[id]._results.size(); i++) {
                for (size_t j = 0; j < ICompiledModel::outputs().size(); j++)
                    if (orderedSubgraphs[id]._results[i] == orig_results[j])
                        m_outputs_to_submodels_outputs[j] = {id, i};
            }
        }

        // Prepare mapping between manually splitted inputs/outputs
        // to connect tensors between compiled submodels
        for (const auto& kvp : subgraphParameterToPrevResult) {
            const auto& intermed_output = kvp.second;
            const auto& intermed_input = kvp.first;
            for (size_t id = 0; id < orderedSubgraphs.size(); id++) {
                const auto& out_it = std::find(orderedSubgraphs[id]._results.begin(),
                                               orderedSubgraphs[id]._results.end(),
                                               intermed_output);
                if (out_it != orderedSubgraphs[id]._results.end()) {
                    for (size_t id2 = 0; id2 < orderedSubgraphs.size(); id2++) {
                        if (id2 == id)
                            continue;
                        const auto& in_it = std::find(orderedSubgraphs[id2]._parameters.begin(),
                                                      orderedSubgraphs[id2]._parameters.end(),
                                                      intermed_input);
                        if (in_it != orderedSubgraphs[id2]._parameters.end()) {
                            auto out_idx = std::distance(orderedSubgraphs[id]._results.begin(), out_it);
                            auto in_idx = std::distance(orderedSubgraphs[id2]._parameters.begin(), in_it);
                            m_submodels_input_to_prev_output[{id2, in_idx}] = {id, out_idx};
                        }
                    }
                }
            }
        }

        m_compiled_submodels.resize(orderedSubgraphs.size());
        std::vector<std::shared_ptr<ov::Model>> subFunctions(orderedSubgraphs.size());
        int id = 0;
        for (auto&& subgraph : orderedSubgraphs) {
            m_compiled_submodels[id].device = subgraph._affinity;
            subFunctions[id] = std::make_shared<ov::Model>(subgraph._results,
                                                           subgraph._sinks,
                                                           subgraph._parameters,
                                                           m_name + '_' + std::to_string(id));
            m_compiled_submodels[id].model = subFunctions[id]->clone();  // TODO vurusovs IS CLONE REQUIRED?

            auto metaDevices = get_hetero_plugin()->get_properties_per_device(m_compiled_submodels[id].device,
                                                                              m_cfg.GetDeviceProperties());

            // disable caching for subgraphs, because the whole HETERO model is cached
            auto device_config = metaDevices[m_compiled_submodels[id].device];
            device_config[ov::cache_dir.name()] = "";

            m_compiled_submodels[id].compiled_model = plugin->get_core()->compile_model(m_compiled_submodels[id].model,
                                                                                        m_compiled_submodels[id].device,
                                                                                        device_config);
            ++id;
        }

        // Restore inputs/outputs from compiled models
        m_compiled_inputs.reserve(m_inputs_to_submodels_inputs.size());
        for (const auto& it : m_inputs_to_submodels_inputs) {
            const auto& submodel_idx = it.first;
            const auto& input_idx = it.second;
            m_compiled_inputs.emplace_back(m_compiled_submodels[submodel_idx].compiled_model->inputs()[input_idx]);
        }
        m_compiled_outputs.reserve(m_outputs_to_submodels_outputs.size());
        for (const auto& it : m_outputs_to_submodels_outputs) {
            const auto& submodel_idx = it.first;
            const auto& output_idx = it.second;
            m_compiled_outputs.emplace_back(m_compiled_submodels[submodel_idx].compiled_model->outputs()[output_idx]);
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
        OPENVINO_THROW("Fail to read Hetero device xml header");
    }

    using namespace pugixml::utils;

    pugi::xml_node heteroNode = heteroXmlDoc.document_element();
    m_name = GetStrAttr(heteroNode, "name");

    ov::AnyMap properties;
    auto heteroConfigsNode = heteroNode.child("hetero_config");
    FOREACH_CHILD (heteroConfigNode, heteroConfigsNode, "config") {
        properties.emplace(GetStrAttr(heteroConfigNode, "key"), GetStrAttr(heteroConfigNode, "value"));
    }

    m_cfg = ov::hetero::Configuration(properties, m_cfg);

    pugi::xml_node subnetworksNode = heteroNode.child("compiled_submodels");
    FOREACH_CHILD (subnetworkNode, subnetworksNode, "compiled_submodel") {
        auto device = GetStrAttr(subnetworkNode, "device");

        auto metaDevices = get_hetero_plugin()->get_properties_per_device(device, m_cfg.GetHeteroProperties());
        assert(metaDevices.size() == 1);
        auto& loadConfig = metaDevices[device];

        ov::SoPtr<ov::ICompiledModel> compiled_model;
        std::shared_ptr<ov::Model> ov_model;

        if (get_plugin()->get_core()->device_supports_model_caching(device)) {
            compiled_model = plugin->get_core()->import_model(model, device, loadConfig);
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

            ov_model = plugin->get_core()->read_model(xmlString, weights);
            compiled_model = plugin->get_core()->compile_model(ov_model, device, loadConfig);
        }

        m_compiled_submodels.emplace_back(ov::hetero::CompiledModel::CompiledModelDesc{
            device,
            ov_model,
            compiled_model,
        });
    }

    auto inputs_map_node = heteroNode.child("inputs_to_submodels_inputs");
    FOREACH_CHILD (xml_node, inputs_map_node, "pair") {
        m_inputs_to_submodels_inputs.emplace_back(GetUInt64Attr(xml_node, "submodel_idx"),
                                                  GetUInt64Attr(xml_node, "node_idx"));
    }
    auto outputs_map_node = heteroNode.child("outputs_to_submodels_outputs");
    FOREACH_CHILD (xml_node, outputs_map_node, "pair") {
        m_outputs_to_submodels_outputs.emplace_back(GetUInt64Attr(xml_node, "submodel_idx"),
                                                    GetUInt64Attr(xml_node, "node_idx"));
    }
    auto submodels_input_to_prev_output_node = heteroNode.child("submodels_input_to_prev_output");
    FOREACH_CHILD (xml_node, submodels_input_to_prev_output_node, "record") {
        std::pair<uint64_t, uint64_t> in_pair = {GetUInt64Attr(xml_node, "in_submodel_idx"),
                                                 GetUInt64Attr(xml_node, "in_node_idx")};
        std::pair<uint64_t, uint64_t> out_pair = {GetUInt64Attr(xml_node, "out_submodel_idx"),
                                                  GetUInt64Attr(xml_node, "out_node_idx")};
        m_submodels_input_to_prev_output.emplace(in_pair, out_pair);
    }

    // Restore inputs/outputs from compiled models
    m_compiled_inputs.reserve(m_inputs_to_submodels_inputs.size());
    for (const auto& it : m_inputs_to_submodels_inputs) {
        const auto& submodel_idx = it.first;
        const auto& input_idx = it.second;
        m_compiled_inputs.emplace_back(m_compiled_submodels[submodel_idx].compiled_model->inputs()[input_idx]);
    }
    m_compiled_outputs.reserve(m_outputs_to_submodels_outputs.size());
    for (const auto& it : m_outputs_to_submodels_outputs) {
        const auto& submodel_idx = it.first;
        const auto& output_idx = it.second;
        m_compiled_outputs.emplace_back(m_compiled_submodels[submodel_idx].compiled_model->outputs()[output_idx]);
    }
}

std::shared_ptr<ov::ISyncInferRequest> ov::hetero::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<ov::hetero::InferRequest>(
        std::static_pointer_cast<const ov::hetero::CompiledModel>(shared_from_this()));
}

std::shared_ptr<ov::IAsyncInferRequest> ov::hetero::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<ov::hetero::AsyncInferRequest>(
        std::static_pointer_cast<ov::hetero::InferRequest>(internal_request),
        get_task_executor(),
        get_callback_executor());

    return async_infer_request;
}

void ov::hetero::CompiledModel::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}

std::shared_ptr<const ov::Model> ov::hetero::CompiledModel::get_runtime_model() const {
    OPENVINO_NOT_IMPLEMENTED;
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
        for (auto&& comp_model_desc : m_compiled_submodels) {
            ov::AnyMap device_properties = {};
            if (all_devices.count(comp_model_desc.device) == 0) {
                auto device_supported_metrics =
                    comp_model_desc.compiled_model->get_property(METRIC_KEY(SUPPORTED_METRICS));
                for (auto&& property_name : device_supported_metrics.as<std::vector<std::string>>()) {
                    device_properties[property_name] = comp_model_desc.compiled_model->get_property(property_name);
                }
                auto device_supported_configs =
                    comp_model_desc.compiled_model->get_property(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                for (auto&& property_name : device_supported_configs.as<std::vector<std::string>>()) {
                    device_properties[property_name] = comp_model_desc.compiled_model->get_property(property_name);
                }
                all_devices[comp_model_desc.device] = device_properties;
            }
        }
        return all_devices;
    } else if (ov::model_name == name) {
        return decltype(ov::model_name)::value_type(m_name);
    } else if (ov::loaded_from_cache == name) {
        return decltype(ov::loaded_from_cache)::value_type{m_loaded_from_cache};
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = 0u;
        for (auto&& comp_model_desc : m_compiled_submodels) {
            value = std::max(value,
                             comp_model_desc.compiled_model->get_property(ov::optimal_number_of_infer_requests.name())
                                 .as<unsigned int>());
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type{value};
    } else if (ov::execution_devices == name) {
        std::vector<std::string> device_names;
        std::set<std::string> s;
        for (auto&& comp_model_desc : m_compiled_submodels) {
            if (s.count(comp_model_desc.device) != 0)
                continue;
            s.insert(comp_model_desc.device);
            device_names.push_back(comp_model_desc.device);
        }
        return decltype(ov::execution_devices)::value_type{device_names};
    }
    return m_cfg.Get(name);
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::inputs() const {
    return m_compiled_inputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::outputs() const {
    return m_compiled_outputs;
}

void ov::hetero::CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "CompiledModel::export_model");

    pugi::xml_document doc;
    auto heteroNode = doc.append_child("hetero");
    heteroNode.append_attribute("name").set_value(m_name.c_str());

    auto inputs_map_node = heteroNode.append_child("inputs_to_submodels_inputs");
    for (auto&& it : m_inputs_to_submodels_inputs) {
        auto xml_node = inputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(it.first);
        xml_node.append_attribute("node_idx").set_value(it.second);
    }
    auto outputs_map_node = heteroNode.append_child("outputs_to_submodels_outputs");
    for (auto&& it : m_outputs_to_submodels_outputs) {
        auto xml_node = outputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(it.first);
        xml_node.append_attribute("node_idx").set_value(it.second);
    }

    auto submodels_input_to_prev_output_node = heteroNode.append_child("submodels_input_to_prev_output");
    for (auto&& it : m_submodels_input_to_prev_output) {
        auto xml_node = submodels_input_to_prev_output_node.append_child("record");
        xml_node.append_attribute("in_submodel_idx").set_value(it.first.first);
        xml_node.append_attribute("in_node_idx").set_value(it.first.second);
        xml_node.append_attribute("out_submodel_idx").set_value(it.second.first);
        xml_node.append_attribute("out_node_idx").set_value(it.second.second);
    }

    auto subnetworksNode = heteroNode.append_child("compiled_submodels");
    for (auto&& comp_model_desc : m_compiled_submodels) {
        auto sub_comp_model = comp_model_desc.compiled_model;
        OPENVINO_ASSERT(sub_comp_model);

        auto subnetworkNode = subnetworksNode.append_child("compiled_submodel");
        subnetworkNode.append_attribute("device").set_value(comp_model_desc.device.c_str());
    }

    auto heteroConfigsNode = heteroNode.append_child("hetero_config");
    for (auto&& config : m_cfg.GetHeteroProperties()) {
        auto heteroConfigNode = heteroConfigsNode.append_child("config");
        heteroConfigNode.append_attribute("key").set_value(config.first.c_str());
        heteroConfigNode.append_attribute("value").set_value(config.second.as<std::string>().c_str());
    }

    doc.save(model_stream, nullptr, pugi::format_raw);
    doc.reset();
    model_stream << std::endl;

    for (auto&& comp_model_desc : m_compiled_submodels) {
        if (get_plugin()->get_core()->device_supports_model_caching(comp_model_desc.device)) {
            comp_model_desc.compiled_model->export_model(model_stream);
        } else {
            auto model = comp_model_desc.model;
            if (!model)
                OPENVINO_THROW("Hetero device supports only OpenVINO Model representation");

            std::stringstream xmlFile, binFile;
            ov::pass::Serialize serializer(xmlFile, binFile);
            serializer.run_on_model(model);

            auto constants = binFile.str();
            auto model_str = xmlFile.str();

            auto dataSize = static_cast<std::uint64_t>(model_str.size());
            model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            model_stream.write(model_str.c_str(), dataSize);

            dataSize = static_cast<std::uint64_t>(constants.size());
            model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            model_stream.write(reinterpret_cast<char*>(&constants[0]), dataSize);
        }
    }
}