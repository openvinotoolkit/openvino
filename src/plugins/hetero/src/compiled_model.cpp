// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "graph_debug_dump.hpp"
#include "ie_plugin_config.hpp"
#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin.hpp"
#include "properties.hpp"
#include "subgraph_collector.hpp"
#include "xml_parse_utils.h"

ov::hetero::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg)
    : ov::ICompiledModel(model, plugin),
      m_cfg(cfg),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(false) {
    try {
        compile_model(model);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Standard exception from compilation library: ", e.what());
    } catch (...) {
        OPENVINO_THROW("Generic exception is thrown");
    }
}

void ov::hetero::CompiledModel::compile_model(const std::shared_ptr<ov::Model>& model) {
    bool dump_dot_file = m_cfg.dump_graph;
    if (std::getenv("OPENVINO_HETERO_VISUALIZE"))
        dump_dot_file = true;

    // Calling of ConstantFolding in HETERO plugin is required because
    // in some cases topology split is happening after constant subgraph.
    // It may cause replacement of Constant by Parameter in such operations
    // like Reshape/Transpose/Gather and lead to unexpected dynamism or exception
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.run_passes(model);

    ov::SupportedOpsMap query_model_result;
    auto ordered_ops = model->get_ordered_ops();

    bool all_empty = true;
    // Get user defined affinity
    for (const auto& node : ordered_ops) {
        auto& node_info = node->get_rt_info();
        auto it_info = node_info.find("affinity");
        if (it_info != node_info.end()) {
            OPENVINO_ASSERT(it_info->second.is<std::string>(), "Unexpected type of \"affinity\" attribute");
            query_model_result.emplace(node->get_friendly_name(), it_info->second.as<std::string>());
            all_empty = false;
        }
    }

    if (query_model_result.empty()) {
        // Restore properties in order to pass "device priorities" together
        // with devices properties
        auto full_properties = m_cfg.get_hetero_properties();
        for (const auto& property : m_cfg.get_device_properties())
            full_properties[property.first] = property.second;
        query_model_result = get_hetero_plugin()->query_model(model, full_properties);
    }

    std::unordered_set<std::string> devices;
    SubgraphCollector::AffinitiesMap affinities;
    // Check that all nodes has user or plugin defined affinities
    for (const auto& node : ordered_ops) {
        auto it_affinity = query_model_result.find(node->get_friendly_name());
        if (it_affinity != query_model_result.end()) {
            affinities[node] = it_affinity->second;
            devices.emplace(it_affinity->second);
        } else if (all_empty) {
            OPENVINO_THROW("Hetero device used default fallback policy, but some layers eg: \n(Name:",
                           node->get_friendly_name(),
                           ", Type: ",
                           node->get_type_name(),
                           ") were not able to be assigned on any pointed device.\n",
                           "It happened because these layers are not supported in plugins by default.\n",
                           "You need to implement custom layers to support them.");
        } else {
            OPENVINO_THROW("Model passed to CompiledModel has affinity assigned, but some layers eg: \n(Name:",
                           node->get_friendly_name(),
                           ", Type: ",
                           node->get_type_name(),
                           ") were not assigned to any device.\n",
                           "It might happen if you assigned layers manually and missed some layers or\n",
                           "if you used some automatic assigning mode which decided that these layers are not\n",
                           "supported by any plugin");
        }
    }

    if (dump_dot_file) {
        ov::hetero::debug::dump_affinities(model, query_model_result, devices);
    }

    // Init subgraph collector
    SubgraphCollector subgraph_collector(model, affinities);

    if (dump_dot_file) {
        auto subgraph_ids = subgraph_collector.get_subgraph_ids();
        std::map<std::string, SubgraphCollector::SubgraphId> map_id;
        for (const auto& v : subgraph_ids) {
            map_id.emplace(v.first->get_friendly_name(), v.second);
        }
        ov::hetero::debug::dump_subgraphs(model, query_model_result, map_id);
    }

    // Get subgraphs sorted topologically
    auto ordered_subgraphs = subgraph_collector.get_ordered_subgraphs();

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
    for (size_t id = 0; id < ordered_subgraphs.size(); id++) {
        for (size_t i = 0; i < ordered_subgraphs[id]._parameters.size(); i++) {
            for (size_t j = 0; j < orig_parameters.size(); j++)
                if (ordered_subgraphs[id]._parameters[i] == orig_parameters[j])
                    m_inputs_to_submodels_inputs[j] = {id, i};
        }
        for (size_t i = 0; i < ordered_subgraphs[id]._results.size(); i++) {
            for (size_t j = 0; j < orig_results.size(); j++)
                if (ordered_subgraphs[id]._results[i] == orig_results[j])
                    m_outputs_to_submodels_outputs[j] = {id, i};
        }
    }

    // Prepare mapping between manually splitted inputs/outputs
    // to connect tensors between compiled submodels
    for (const auto& kvp : subgraph_collector.get_subgraph_parameter_to_prev_result()) {
        const auto& intermed_output = kvp.second;
        const auto& intermed_input = kvp.first;
        for (size_t id = 0; id < ordered_subgraphs.size(); id++) {
            const auto& out_it = std::find(ordered_subgraphs[id]._results.begin(),
                                           ordered_subgraphs[id]._results.end(),
                                           intermed_output);
            if (out_it != ordered_subgraphs[id]._results.end()) {
                for (size_t id2 = 0; id2 < ordered_subgraphs.size(); id2++) {
                    if (id2 == id)
                        continue;
                    const auto& in_it = std::find(ordered_subgraphs[id2]._parameters.begin(),
                                                  ordered_subgraphs[id2]._parameters.end(),
                                                  intermed_input);
                    if (in_it != ordered_subgraphs[id2]._parameters.end()) {
                        auto out_idx = std::distance(ordered_subgraphs[id]._results.begin(), out_it);
                        auto in_idx = std::distance(ordered_subgraphs[id2]._parameters.begin(), in_it);
                        m_submodels_input_to_prev_output[{id2, in_idx}] = {id, out_idx};
                    }
                }
            }
        }
    }

    m_compiled_submodels.resize(ordered_subgraphs.size());
    std::vector<std::shared_ptr<ov::Model>> submodels(ordered_subgraphs.size());
    size_t id = 0;
    for (const auto& subgraph : ordered_subgraphs) {
        m_compiled_submodels[id].device = subgraph._affinity;
        submodels[id] = std::make_shared<ov::Model>(subgraph._results,
                                                    subgraph._sinks,
                                                    subgraph._parameters,
                                                    m_name + '_' + std::to_string(id));
        m_compiled_submodels[id].model = submodels[id];

        auto meta_devices = get_hetero_plugin()->get_properties_per_device(m_compiled_submodels[id].device,
                                                                           m_cfg.get_device_properties());

        // disable caching for subgraphs, because the whole HETERO model is cached
        auto device_config = meta_devices[m_compiled_submodels[id].device];
        device_config[ov::cache_dir.name()] = "";
        // set exclusive_async_requests in case when model is split
        if (ordered_subgraphs.size() > 1) {
            auto supported_internal_properties =
                get_hetero_plugin()->get_core()->get_property(m_compiled_submodels[id].device,
                                                              ov::internal::supported_properties);
            if (std::find(supported_internal_properties.begin(),
                          supported_internal_properties.end(),
                          ov::internal::exclusive_async_requests) != supported_internal_properties.end()) {
                // adds property if it is not set yet
                device_config.insert(ov::internal::exclusive_async_requests(true));
            }
        }
        m_compiled_submodels[id].compiled_model =
            get_hetero_plugin()->get_core()->compile_model(m_compiled_submodels[id].model,
                                                           m_compiled_submodels[id].device,
                                                           device_config);
        ++id;
    }

    set_inputs_and_outputs();
}

ov::hetero::CompiledModel::CompiledModel(std::istream& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg)
    : ov::ICompiledModel(nullptr, plugin),
      m_cfg(cfg),
      m_name(),
      m_loaded_from_cache(true) {
    std::string heteroXmlStr;
    std::getline(model, heteroXmlStr);

    pugi::xml_document heteroXmlDoc;
    pugi::xml_parse_result res = heteroXmlDoc.load_string(heteroXmlStr.c_str());

    if (res.status != pugi::status_ok)
        OPENVINO_THROW("Failed to read Hetero device xml header");

    using namespace pugixml::utils;

    pugi::xml_node heteroNode = heteroXmlDoc.document_element();
    m_name = GetStrAttr(heteroNode, "name");

    ov::AnyMap properties;
    auto heteroConfigsNode = heteroNode.child("hetero_config");
    FOREACH_CHILD(heteroConfigNode, heteroConfigsNode, "config") {
        properties.emplace(GetStrAttr(heteroConfigNode, "key"), GetStrAttr(heteroConfigNode, "value"));
    }

    m_cfg = ov::hetero::Configuration(properties, m_cfg);

    pugi::xml_node subnetworksNode = heteroNode.child("compiled_submodels");
    FOREACH_CHILD(subnetworkNode, subnetworksNode, "compiled_submodel") {
        auto device = GetStrAttr(subnetworkNode, "device");

        auto meta_devices = get_hetero_plugin()->get_properties_per_device(device, m_cfg.get_device_properties());
        assert(meta_devices.size() == 1);
        auto& loadConfig = meta_devices[device];

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
    FOREACH_CHILD(xml_node, inputs_map_node, "pair") {
        m_inputs_to_submodels_inputs.emplace_back(GetUInt64Attr(xml_node, "submodel_idx"),
                                                  GetUInt64Attr(xml_node, "node_idx"));
    }
    auto outputs_map_node = heteroNode.child("outputs_to_submodels_outputs");
    FOREACH_CHILD(xml_node, outputs_map_node, "pair") {
        m_outputs_to_submodels_outputs.emplace_back(GetUInt64Attr(xml_node, "submodel_idx"),
                                                    GetUInt64Attr(xml_node, "node_idx"));
    }
    auto submodels_input_to_prev_output_node = heteroNode.child("submodels_input_to_prev_output");
    FOREACH_CHILD(xml_node, submodels_input_to_prev_output_node, "record") {
        std::pair<uint64_t, uint64_t> in_pair = {GetUInt64Attr(xml_node, "in_submodel_idx"),
                                                 GetUInt64Attr(xml_node, "in_node_idx")};
        std::pair<uint64_t, uint64_t> out_pair = {GetUInt64Attr(xml_node, "out_submodel_idx"),
                                                  GetUInt64Attr(xml_node, "out_node_idx")};
        m_submodels_input_to_prev_output.emplace(in_pair, out_pair);
    }
    set_inputs_and_outputs();
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
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> ov::hetero::CompiledModel::get_runtime_model() const {
    std::vector<std::shared_ptr<ov::Model>> rt_models;
    // Collect runtime subgraphs
    for (size_t i = 0; i < m_compiled_submodels.size(); i++) {
        rt_models.push_back(m_compiled_submodels.at(i).compiled_model->get_runtime_model()->clone());
    }
    // Results which should not be present in final graph
    std::set<std::string> result_names_to_be_removed;
    // Remap port indexes to names, because order of them will be modified during merge
    std::map<std::pair<size_t, std::string>, std::pair<size_t, std::string>> input_to_prev_output;
    for (const auto& kvp : m_submodels_input_to_prev_output) {
        const auto& input_node = rt_models[kvp.first.first]->inputs()[kvp.first.second].get_node();
        const auto& output_node = rt_models[kvp.second.first]->outputs()[kvp.second.second].get_node();
        input_to_prev_output[{kvp.first.first, input_node->get_friendly_name()}] = {kvp.second.first,
                                                                                    output_node->get_friendly_name()};
        result_names_to_be_removed.insert(output_node->get_friendly_name());
    }
    int submodel_in_index = static_cast<int>(rt_models.size()) - 1;
    while (submodel_in_index >= 0 && input_to_prev_output.size() > 0) {
        auto& submodel_in = rt_models[submodel_in_index];
        size_t port_in_index = 0;
        while (port_in_index < submodel_in->get_parameters().size()) {
            auto parameter_to_replace = submodel_in->get_parameters()[port_in_index];
            auto item = input_to_prev_output.find({submodel_in_index, parameter_to_replace->get_friendly_name()});
            if (item == input_to_prev_output.end()) {
                port_in_index++;
                continue;
            }
            auto submodel_out_index = item->second.first;
            auto submodel_out_result_name = item->second.second;
            auto submodel_out = rt_models.at(submodel_out_index);

            // Get all results from previous subgraph except already existed in next subgraph
            std::shared_ptr<ov::op::v0::Result> result_to_replace = nullptr;
            ov::ResultVector add_results;
            for (auto& result : submodel_out->get_results()) {
                if (result->get_friendly_name() == submodel_out_result_name) {
                    result_to_replace = result;
                }
                auto it = std::find_if(submodel_in->get_results().begin(),
                                       submodel_in->get_results().end(),
                                       [&](const std::shared_ptr<ov::op::v0::Result>& result_to_check) {
                                           return result_to_check == result;
                                       });
                if (it == submodel_in->get_results().end())
                    add_results.push_back(result);
            }
            OPENVINO_ASSERT(result_to_replace != nullptr);

            // Get all parameters from previous subgraph except already existed in next subgraph
            ov::ParameterVector add_parameters;
            for (auto& parameter : submodel_out->get_parameters()) {
                auto it = std::find_if(submodel_in->get_parameters().begin(),
                                       submodel_in->get_parameters().end(),
                                       [&](const std::shared_ptr<ov::op::v0::Parameter>& parameter_to_check) {
                                           return parameter_to_check == parameter;
                                       });
                if (it == submodel_in->get_parameters().end())
                    add_parameters.push_back(parameter);
            }

            // Reconnect appropariate target inputs to the new source output
            auto result_source = result_to_replace->get_input_source_output(0);
            auto parameter_targets = parameter_to_replace->get_output_target_inputs(0);
            for (auto parameter_target : parameter_targets) {
                parameter_target.replace_source_output(result_source);
            }

            // Update parameter and results
            submodel_in->remove_parameter(parameter_to_replace);
            submodel_in->add_parameters(add_parameters);
            submodel_in->add_results(add_results);

            // Remove processed connection
            input_to_prev_output.erase(item);

            // Update incoming model since it is merged
            for (size_t i = 0; i < rt_models.size(); i++) {
                if (rt_models[i] == submodel_out) {
                    rt_models[i] = submodel_in;
                }
            }

            // Start check ports from the beginning because number of ports are modified
            port_in_index = 0;
        }
        --submodel_in_index;
    }
    // Finally all subgraphs should be merged into single one
    OPENVINO_ASSERT(input_to_prev_output.size() == 0);
    OPENVINO_ASSERT(all_of(rt_models.begin(), rt_models.end(), [&](const std::shared_ptr<ov::Model>& rt_model) {
        return rt_model == rt_models[0];
    }));
    auto runtime_graph = rt_models[0];
    // Cleanup intermidiate results
    for (size_t i = 0; i < runtime_graph->get_results().size();) {
        auto& result = runtime_graph->get_results()[i];
        if (result_names_to_be_removed.count(result->get_friendly_name())) {
            runtime_graph->remove_result(result);
        } else {
            i++;
        }
    }
    OPENVINO_ASSERT(runtime_graph->inputs().size() == inputs().size());
    return runtime_graph;
}

std::shared_ptr<const ov::hetero::Plugin> ov::hetero::CompiledModel::get_hetero_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto hetero_plugin = std::static_pointer_cast<const ov::hetero::Plugin>(plugin);
    OPENVINO_ASSERT(hetero_plugin);
    return hetero_plugin;
}

ov::Any ov::hetero::CompiledModel::get_property(const std::string& name) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache,
                                                    ov::hetero::number_of_submodels};
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
        return to_string_vector(m_cfg.get_supported());
    } else if (ov::device::properties == name) {
        ov::AnyMap all_devices = {};
        for (const auto& comp_model_desc : m_compiled_submodels) {
            ov::AnyMap device_properties = {};
            if (all_devices.count(comp_model_desc.device) == 0) {
                auto device_supported_props =
                    comp_model_desc.compiled_model->get_property(ov::supported_properties.name());
                for (auto&& property_name : device_supported_props.as<std::vector<ov::PropertyName>>())
                    device_properties[property_name] = comp_model_desc.compiled_model->get_property(property_name);
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
        for (const auto& comp_model_desc : m_compiled_submodels) {
            value = std::max(value,
                             comp_model_desc.compiled_model->get_property(ov::optimal_number_of_infer_requests.name())
                                 .as<unsigned int>());
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type{value};
    } else if (ov::execution_devices == name) {
        std::vector<std::string> device_names;
        std::set<std::string> s;
        for (const auto& comp_model_desc : m_compiled_submodels) {
            if (s.count(comp_model_desc.device) != 0)
                continue;
            s.insert(comp_model_desc.device);
            device_names.push_back(comp_model_desc.device);
        }
        return decltype(ov::execution_devices)::value_type{device_names};
    } else if (ov::hetero::number_of_submodels == name) {
        return decltype(ov::hetero::number_of_submodels)::value_type{m_compiled_submodels.size()};
    }
    return m_cfg.get(name);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::inputs() const {
    return m_compiled_inputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::outputs() const {
    return m_compiled_outputs;
}

void ov::hetero::CompiledModel::set_inputs_and_outputs() {
    // Restore inputs/outputs from compiled submodels
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

void ov::hetero::CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "CompiledModel::export_model");

    pugi::xml_document doc;
    auto heteroNode = doc.append_child("hetero");
    heteroNode.append_attribute("name").set_value(m_name.c_str());

    auto inputs_map_node = heteroNode.append_child("inputs_to_submodels_inputs");
    for (const auto& it : m_inputs_to_submodels_inputs) {
        auto xml_node = inputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(std::to_string(it.first).c_str());
        xml_node.append_attribute("node_idx").set_value(std::to_string(it.second).c_str());
    }
    auto outputs_map_node = heteroNode.append_child("outputs_to_submodels_outputs");
    for (const auto& it : m_outputs_to_submodels_outputs) {
        auto xml_node = outputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(std::to_string(it.first).c_str());
        xml_node.append_attribute("node_idx").set_value(std::to_string(it.second).c_str());
    }

    auto submodels_input_to_prev_output_node = heteroNode.append_child("submodels_input_to_prev_output");
    for (const auto& it : m_submodels_input_to_prev_output) {
        auto xml_node = submodels_input_to_prev_output_node.append_child("record");
        xml_node.append_attribute("in_submodel_idx").set_value(std::to_string(it.first.first).c_str());
        xml_node.append_attribute("in_node_idx").set_value(std::to_string(it.first.second).c_str());
        xml_node.append_attribute("out_submodel_idx").set_value(std::to_string(it.second.first).c_str());
        xml_node.append_attribute("out_node_idx").set_value(std::to_string(it.second.second).c_str());
    }

    auto subnetworksNode = heteroNode.append_child("compiled_submodels");
    for (const auto& comp_model_desc : m_compiled_submodels) {
        auto sub_comp_model = comp_model_desc.compiled_model;
        OPENVINO_ASSERT(sub_comp_model);

        auto subnetworkNode = subnetworksNode.append_child("compiled_submodel");
        subnetworkNode.append_attribute("device").set_value(comp_model_desc.device.c_str());
    }

    auto heteroConfigsNode = heteroNode.append_child("hetero_config");
    for (const auto& config : m_cfg.get_hetero_properties()) {
        auto heteroConfigNode = heteroConfigsNode.append_child("config");
        heteroConfigNode.append_attribute("key").set_value(config.first.c_str());
        heteroConfigNode.append_attribute("value").set_value(config.second.as<std::string>().c_str());
    }

    doc.save(model_stream, nullptr, pugi::format_raw);
    doc.reset();
    model_stream << std::endl;

    for (const auto& comp_model_desc : m_compiled_submodels) {
        if (get_plugin()->get_core()->device_supports_model_caching(comp_model_desc.device)) {
            try {
                // Batch plugin reports property of low level plugin
                // If we use Batch plugin inside hetero, we won't be able to call export
                // Auto batch plugin will throw NOT_IMPLEMENTED
                comp_model_desc.compiled_model->export_model(model_stream);
                continue;
            } catch (ov::NotImplemented&) {
            }
        }
        auto model = comp_model_desc.model;
        if (!model)
            OPENVINO_THROW("OpenVINO Model is empty");

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
