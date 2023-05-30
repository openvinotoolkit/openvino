// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "plugin.hpp"
#include "async_infer_request.hpp"

#include "itt.hpp"

// #include "perf_counter.hpp"
// #include "graph_debug_dump.hpp"


#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"


#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "ie_algorithm.hpp"

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
                IE_ASSERT(itInfo->second.is<std::string>());
                queryNetworkResult.emplace(node->get_friendly_name(), itInfo->second.as<std::string>());
                allEmpty = false;
            }
        }

        if (queryNetworkResult.empty()) {
            // here we need to bypass unchanged / unparsed user-set configuration
            // because it can contain TARGET_FALLBACK / ov::device::priorities
            queryNetworkResult = plugin->query_model(model, {ov::device::priorities(m_cfg.device_priorities)}); // TODO vurusovs DECIDE ABOUT HETERO m_cfg.GetHeteroConfig()
        }

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

        // if (dumpDotFile) {
        //     ov::hetero::debug::dump_affinities(model, queryNetworkResult, devices);
        // }

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
                    if (!InferenceEngine::details::contains(subgraphInputs, input)) { // TODO vurusovs REPLACE with ov::util::contains
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
            IE_ASSERT(cyclicSplitStep < orderedOps.size());
            prevSubgraphs = subgraphInputs.size();
            auto subgraphIds = CollectSubgraphs();
            // All inputs that belong to the same subgraph as node
            std::unordered_map<ov::Node*, InputSet> nodeSubgraphInputDependencies;
            // All inputs that depends on the same subgraph as node
            std::unordered_map<ov::Node*, InputSet> nodeSubgraphCyclicInputDependencies;
            for (auto&& node : orderedOps) {
                auto& nodeSubgraphInputDependency = nodeSubgraphInputDependencies[node.get()];
                auto allNodeSubgraphInputs = InferenceEngine::details::Intersection(nodeInputDependencies[node.get()], subgraphInputs);
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
                    if (!InferenceEngine::details::Intersects(nodeSubgraphCyclicInputDependency, inputNodeSubgraphCyclicInputDependency) &&
                        InferenceEngine::details::Intersects(cyclicInputsDependencies, inputNodeSubgraphInputDependency)) {
                        subgraphInputs.insert(input);
                    }
                }
            }
        }
    }

    auto subgraphIds = CollectSubgraphs();

    // if (dumpDotFile) {
    //     std::map<std::string, int> map_id;
    //     for (auto&& v : subgraphIds) {
    //         map_id.emplace(v.first->get_friendly_name(), v.second);
    //     }
    //     ov::hetero::debug::dump_subgraphs(std::const_pointer_cast<ov::Model>(function),
    //                                       queryNetworkResult.supportedLayersMap,
    //                                       map_id);
    // }

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
                    auto parameter =
                        std::make_shared<ngraph::op::Parameter>(output.get_element_type(), output.get_partial_shape());
                    parameter->set_friendly_name(input.get_node()->get_friendly_name() + "_" +
                                                 std::to_string(input.get_index()) + "_parameter");
                    ov::copy_runtime_info(input.get_node()->shared_from_this(), parameter);
                    input.replace_source_output(parameter->output(0));
                    subgraphIds.emplace(parameter.get(), input_subset.first);
                    subgraphParameterToPrevResult.emplace(parameter.get(), result.get());
                    _blobNameMap.emplace(parameter->get_default_output(), output);
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
            subgraph._results.emplace_back(std::dynamic_pointer_cast<ngraph::op::v0::Result>(node->shared_from_this()));
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
        IE_ASSERT(subgraphTopoSortsStep < subgraphs.size());
        ++subgraphTopoSortsStep;
        std::vector<Subgraph> newOrderedSubgraphs;
        auto IsOrderedSubGraph = [&](const Subgraph& subgraph) {
            auto& parameters = subgraph._parameters;
            return std::all_of(parameters.begin(),
                               parameters.end(),
                               [&](const ngraph::ParameterVector::value_type& parameter) {
                                   return InferenceEngine::details::contains(graphInputNodes, parameter.get()) ||
                                          InferenceEngine::details::contains(prevResults, subgraphParameterToPrevResult[parameter.get()]);
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
        std::move(std::begin(newOrderedSubgraphs), std::end(newOrderedSubgraphs), std::back_inserter(orderedSubgraphs));
    } while (!allSubgraphs.empty());

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
        m_networks[id]._clonedNetwork = subFunctions[id]->clone(); // TODO vurusovs IS CLONE REQUIRED? 
        
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
        //     return (ngraph_type == ngraph::element::f16 || ngraph_type == ngraph::element::bf16) ? ngraph::element::f32
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
        auto metaDevices = get_hetero_plugin()->GetDevicePlugins(network._device, m_cfg.GetDeviceConfig());

        // disable caching for subgraphs, because the whole HETERO model is cached
        auto device_config = metaDevices[network._device];
        device_config[ov::cache_dir.name()] = "";

        network._network = plugin->get_core()->compile_model(network._clonedNetwork, network._device, device_config);
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

std::shared_ptr<ov::ISyncInferRequest> ov::hetero::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<ov::hetero::InferRequest>(
        std::static_pointer_cast<const ov::hetero::CompiledModel>(shared_from_this()));
    return nullptr;
}

std::shared_ptr<ov::IAsyncInferRequest> ov::hetero::CompiledModel::create_infer_request() const {
    // TODO vurusovs WAIT FOR ov::hetero::AsyncInferRequest and ov::hetero::InferRequest
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

// ! [compiled_model:get_runtime_model]
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
                                                    ov::supported_properties,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache,
                                                    ov::optimal_number_of_infer_requests};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::id, ov::enable_profiling};
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    // TODO: return more supported values for metrics
    if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();
        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        return to_string_vector(metrics);
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        auto configs = default_rw_properties();
        auto streamExecutorConfigKeys = ov::threading::IStreamsExecutor::Config{}
                                            .get_property(ov::supported_properties.name())
                                            .as<std::vector<std::string>>();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configs.emplace_back(configKey);
        }
        return to_string_vector(configs);
    } else if (ov::model_name == name) {
        auto model_name = m_model->get_friendly_name();
        return decltype(ov::model_name)::value_type(model_name);
    } else if (ov::loaded_from_cache == name) {
        return m_loaded_from_cache;
    // } else if (ov::execution_devices == name) {
    //     return decltype(ov::execution_devices)::value_type{get_plugin()->get_device_name() + "." +
    //                                                        std::to_string(m_cfg.device_id)};
    // } else if (ov::optimal_number_of_infer_requests == name) {
    //     unsigned int value = m_cfg.streams_executor_config._streams;
    //     return decltype(ov::optimal_number_of_infer_requests)::value_type(value);
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    }

    return m_cfg.Get(name);
}

void ov::hetero::CompiledModel::export_model(std::ostream& model_stream) const {
    // TODO vurusovs CONTINUE - SPLIT FOR SEVERAL SUBNETWORKS
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "CompiledModel::export_model");

    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, binFile);
    serializer.run_on_model(m_model);

    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();

    auto dataSize = static_cast<std::uint64_t>(m_model.size());
    model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    model_stream.write(m_model.c_str(), dataSize);

    dataSize = static_cast<std::uint64_t>(m_constants.size());
    model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    model_stream.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);
}
