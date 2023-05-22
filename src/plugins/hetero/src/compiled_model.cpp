// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "perf_counter.hpp"
#include "plugin.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

#include "graph_debug_dump.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/common_util.hpp"
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
            plugin->query_model(model, /*TODO vurusovs PROVIDE PROPERTIES*/{});
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
    // TODO vurusovs WAIT FOR ov::hetero::InferRequest
    return std::make_shared<ov::hetero::InferRequest>(
        std::static_pointer_cast<const ov::hetero::CompiledModel>(shared_from_this()));
}

std::shared_ptr<ov::IAsyncInferRequest> ov::hetero::CompiledModel::create_infer_request() const {
    // TODO vurusovs WAIT FOR ov::hetero::AsyncInferRequest and ov::hetero::InferRequest
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<ov::hetero::AsyncInferRequest>(
        std::static_pointer_cast<ov::hetero::InferRequest>(internal_request));

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
        const auto& it = info.find(ov::runtime::interpreter::PERF_COUNTER_NAME);
        OPENVINO_ASSERT(it != info.end(), "Operation ", op, " doesn't contain performance counter");
        auto perf_count = it->second.as<std::shared_ptr<ov::runtime::interpreter::PerfCounter>>();
        OPENVINO_ASSERT(perf_count, "Performance counter is empty");
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
// ! [compiled_model:get_runtime_model]

std::shared_ptr<const ov::hetero::Plugin> ov::hetero::CompiledModel::get_hetero_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto hetero_plugin = std::static_pointer_cast<const ov::hetero::Plugin>(plugin);
    OPENVINO_ASSERT(hetero_plugin);
    return hetero_plugin;
}

// ! [compiled_model:get_property]
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
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_plugin()->get_device_name() + "." +
                                                           std::to_string(m_cfg.device_id)};
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = m_cfg.streams_executor_config._streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(value);
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
// ! [compiled_model:get_property]

// ! [compiled_model:export_model]
void ov::hetero::CompiledModel::export_model(std::ostream& model_stream) const {
    // TODO vurusovs CONTINUE
    OV_ITT_SCOPED_TASK(itt::domains::HeteroPlugin, "CompiledModel::export_model");

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
// ! [compiled_model:export_model]
