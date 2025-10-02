// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/graph.hpp"

#include <exception>
#include <functional>
#include <numeric>
#include <sstream>

#include "core/node.hpp"
#include "core/null_node.hpp"
#include "core/transform.hpp"
#include "core/value_info.hpp"
#include "exceptions.hpp"
#include "onnx_framework_node.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/onnx/extension/conversion.hpp"
#include "openvino/frontend/onnx/node_context.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/common.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/make_stateful.hpp"
#include "openvino/op/gather.hpp"
#include <openvino/core/graph_util.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/pass/constant_folding.hpp>

using namespace ov;
using namespace ::ONNX_NAMESPACE;

namespace ov {
namespace frontend {
namespace onnx {
namespace detail {
bool common_node_for_all_outputs(const ov::OutputVector& outputs) {
    const auto first_out_node = outputs.at(0).get_node();
    bool ret = std::all_of(std::next(std::begin(outputs)),
                           std::end(outputs),
                           [first_out_node](const ov::OutputVector::value_type& output) {
                               return output.get_node() == first_out_node;
                           });
    return ret;
};

OperatorsBridge register_extensions(OperatorsBridge& bridge,
                                    const std::vector<ov::frontend::ConversionExtensionBase::Ptr>& conversions) {
    for (const auto& extension : conversions) {
        if (const auto common_conv_ext = ov::as_type_ptr<ov::frontend::ConversionExtension>(extension)) {
            bridge.overwrite_operator(
                common_conv_ext->get_op_type(),
                "",
                [common_conv_ext](const ov::frontend::onnx::Node& node) -> ov::OutputVector {
                    return common_conv_ext->get_converter()(ov::frontend::onnx::NodeContext(node));
                });
        } else if (const auto onnx_conv_ext = ov::as_type_ptr<ov::frontend::onnx::ConversionExtension>(extension)) {
            bridge.overwrite_operator(onnx_conv_ext->get_op_type(),
                                      onnx_conv_ext->get_domain(),
                                      [onnx_conv_ext](const ov::frontend::onnx::Node& node) -> ov::OutputVector {
                                          return onnx_conv_ext->get_converter()(ov::frontend::onnx::NodeContext(node));
                                      });
        }
    }
    return bridge;
}

OperatorsBridge init_ops_bridge(const std::vector<ov::frontend::ConversionExtensionBase::Ptr>& conversions) {
    OperatorsBridge bridge;
    return register_extensions(bridge, conversions);
}

Model::ModelOpSet build_model_opset(const ModelProto& model_proto, const OperatorsBridge& ops_bridge) {
    // copy the opset imports from the ONNX model and sort them by their version in ascending order
    // this will make sure that multiple opset imports for the same domain will cause the largest
    // version to be used for this model, for example:
    // [{domain:"", version:11}, {domain:"", version:1} {domain:"", version:13}] ==> {domain:"", version:13}
    auto opset_imports = model_proto.opset_import();
    const auto sort_by_version_ascending = [](const OperatorSetIdProto& lhs, const OperatorSetIdProto& rhs) {
        return lhs.version() < rhs.version();
    };
    std::sort(std::begin(opset_imports), std::end(opset_imports), sort_by_version_ascending);

    Model::ModelOpSet opset;
    std::for_each(opset_imports.rbegin(),
                  opset_imports.rend(),
                  [&opset, &ops_bridge](const OperatorSetIdProto& onnx_opset) {
                      const auto domain =
                          onnx_opset.has_domain() ? onnx_opset.domain() == "ai.onnx" ? "" : onnx_opset.domain() : "";
                      if (opset.find(domain) == std::end(opset)) {
                          opset[domain] = ops_bridge.get_operator_set(domain, onnx_opset.version());
                      }
                  });

    // onnx.proto(.3): the empty string ("") for domain or absence of opset_import field
    // implies the operator set that is defined as part of the ONNX specification.
    const auto dm = opset.find("");
    if (dm == std::end(opset)) {
        opset[""] = ops_bridge.get_operator_set("", ONNX_OPSET_VERSION);
    }

    return opset;
}

/// Copies only the extensions required by the Subgraph class.
/// The source is an extension holder retrieved from the parent graph object.
ov::frontend::ExtensionHolder subgraph_required_extensions(
    const ov::frontend::ExtensionHolder& parent_graph_extensions) {
    ov::frontend::ExtensionHolder extensions;
    extensions.telemetry = parent_graph_extensions.telemetry;
    extensions.conversions = parent_graph_extensions.conversions;
    return extensions;
}
}  // namespace detail

Graph::Graph(const std::string& model_dir,
             const std::shared_ptr<ModelProto>& model_proto,
             detail::MappedMemoryHandles mmap_cache,
             ov::frontend::ExtensionHolder extensions)
    : Graph(model_dir, model_proto, common::make_unique<GraphCache>(), mmap_cache, std::move(extensions)) {}

Graph::Graph(const std::string& model_dir,
             const std::shared_ptr<ModelProto>& model_proto,
             std::unique_ptr<GraphCache>&& cache,
             detail::MappedMemoryHandles mmap_cache,
             ov::frontend::ExtensionHolder extensions)
    : m_cache{std::move(cache)},
      m_extensions{std::move(extensions)},
      m_model_dir{model_dir},
      m_mmap_cache{mmap_cache},
      m_ops_bridge{detail::init_ops_bridge(m_extensions.conversions)} {
    m_model = common::make_unique<Model>(model_proto, detail::build_model_opset(*model_proto, m_ops_bridge));

    transform::expand_onnx_functions(*model_proto);

    std::map<std::string, Tensor> initializers;

    // Process all initializers in the graph
    for (const auto& initializer_tensor : m_model->get_graph().initializer()) {
        if (initializer_tensor.has_name()) {
            Tensor tensor = Tensor{initializer_tensor, m_model_dir, m_mmap_cache};
            std::shared_ptr<ov::op::v0::Constant> ov_constant;
            // For each initializer create a Constant node and store it in cache
            try {
                ov_constant = tensor.get_ov_constant();
            } catch (const error::invalid_external_data&) {
                // invalid external data makes initializers creation impossible
                throw;
            } catch (const ov::Exception&) {
                ov_constant = ov::frontend::onnx::common::make_failsafe_constant(tensor.get_ov_type());
            }

            initializers.emplace(initializer_tensor.name(), tensor);
            ov_constant->get_output_tensor(0).set_names({initializer_tensor.name()});
            m_cache->emplace_node(initializer_tensor.name(), std::move(ov_constant));
        }
    }

    // Process all ONNX graph inputs, convert them to OV nodes and store in cache
    for (const auto& input : m_model->get_graph().input()) {
        // Check if a Constant node was already created from an initializer
        if (m_cache->contains(input.name())) {
            continue;
        }

        ValueInfo value_info{input};
        auto ov_node = value_info.get_ov_node(m_parameters, initializers);
        m_cache->emplace_node(input.name(), std::move(ov_node));
    }
}

void Graph::convert_to_ov_nodes() {
    const float total = static_cast<float>(m_model->get_graph().node().size());
    unsigned int completed = 0u;
    std::map<std::string, uint64_t> op_statistics;
    // Process ONNX graph nodes, convert to OV nodes
    for (const auto& node_proto : m_model->get_graph().node()) {
        if (m_extensions.telemetry) {
            std::string op_name =
                (node_proto.has_domain() && node_proto.domain() != ""
                     ? "***." + node_proto.op_type() + "-X"
                     : node_proto.op_type() + +"-" + std::to_string(m_model->get_opset_version(node_proto.domain())));
            op_statistics[op_name]++;
        }
        const Node node{node_proto, this};
        if (!m_model->is_operator_available(node.op_type(), node.domain())) {
            // If a node from an unregistered domain is detected, try registering that domain
            m_model->enable_opset_domain(node.domain(), m_ops_bridge);
        }
        if (node.has_subgraphs()) {
            const auto& subgraphs = node.get_subgraphs();
            for (auto& kv : subgraphs) {
                auto& subgraph = kv.second;
                subgraph->convert();
            }
        }
        ov::OutputVector ov_nodes{make_ov_nodes(node)};
        ++completed;
        m_extensions.progress_reporter->report_progress(completed / total, static_cast<unsigned int>(total), completed);
    }
    if (m_extensions.telemetry) {
        for (const auto& op : op_statistics) {
            m_extensions.telemetry->send_event("op_count", "onnx_" + op.first, static_cast<int>(op.second));
        }
    }
}

bool model_has_input_output_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& input : model->inputs()) {
        if (input.get_any_name() == name)
            return true;
    }
    for (const auto& output : model->outputs()) {
        if (output.get_any_name() == name)
            return true;
    }
    return false;
}

int index_of_model_input_output(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (size_t i = 0; i < model->inputs().size(); i++) {
        if (model->inputs().at(i).get_any_name() == name)
            return i;
    }
    return 0;
}

std::vector<std::shared_ptr<ov::Node>> get_input_nodes(const std::shared_ptr<ov::Node>& node) {
    std::vector<std::shared_ptr<ov::Node>> inputs;
    for (const auto& input : node->inputs()) {
        auto source_node = input.get_source_output().get_node_shared_ptr();
        if (source_node) {
            inputs.push_back(source_node);
        }
    }
    return inputs;
}

std::unordered_set<std::shared_ptr<ov::Node>> find_dependent_nodes(
    const std::shared_ptr<ov::Model>& model,
    const std::unordered_set<std::shared_ptr<ov::Node>>& sources) {
    std::unordered_set<std::shared_ptr<ov::Node>> result = sources;
    for (const auto& node : model->get_ordered_ops()) {
        auto input_nodes = get_input_nodes(node);
        for (const auto& input_node : input_nodes) {
            if (result.count(input_node)) {
                result.insert(node);
                break;
            }
        }
    }
    return result;
}

std::vector<std::shared_ptr<ov::Node>> get_shape_of_ops(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::Node>> result;
    for (const auto& node : model->get_ops()) {
        if (node->get_type_name() == std::string("ShapeOf")) {
            result.push_back(node);
        }
    }
    return result;
}

std::unordered_set<std::shared_ptr<ov::Node>> get_consumer_nodes(const std::shared_ptr<ov::Node>& node) {
    std::unordered_set<std::shared_ptr<ov::Node>> consumers;
    for (const auto& output : node->outputs()) {
        for (const auto& input : output.get_target_inputs()) {
            consumers.insert(input.get_node()->shared_from_this());
        }
    }
    return consumers;
}

std::vector<std::shared_ptr<ov::Node>> get_read_value_ops(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::Node>> result;
    for (const auto& node : model->get_ops()) {
        if (node->get_type_name() == std::string("ReadValue")) {
            result.push_back(node);
        }
    }
    return result;
}

std::vector<std::shared_ptr<ov::Node>> find_output_nodes_of_dependent_subgraph(
    const std::shared_ptr<ov::Model>& model,
    const std::unordered_set<std::shared_ptr<ov::Node>>& sources) {
    // Get "other" inputs: parameters + ReadValue + ShapeOf, excluding sources
    std::unordered_set<std::shared_ptr<ov::Node>> other_inputs(model->get_parameters().begin(),
                                                               model->get_parameters().end());
    for (const auto& node : get_read_value_ops(model)) {
        other_inputs.insert(node);
    }
    for (const auto& node : get_shape_of_ops(model)) {
        other_inputs.insert(node);
    }
    for (const auto& source : sources) {
        other_inputs.erase(source);
    }
    auto other_nodes = find_dependent_nodes(model, other_inputs);
    auto source_dependent_nodes = find_dependent_nodes(model, sources);
    // Subgraph nodes that depend only on sources
    std::unordered_set<std::shared_ptr<ov::Node>> subgraph_nodes;
    for (const auto& node : source_dependent_nodes) {
        if (!other_nodes.count(node)) {
            subgraph_nodes.insert(node);
        }
    }
    // Find nodes on the boundary (with consumers outside the subgraph)
    std::vector<std::shared_ptr<ov::Node>> edge_nodes;
    for (const auto& node : subgraph_nodes) {
        auto consumers = get_consumer_nodes(node);
        for (const auto& consumer : consumers) {
            if (other_nodes.count(consumer)) {
                edge_nodes.push_back(node);
                break;
            }
        }
    }
    return edge_nodes;
}

void insert_state_for_nodes(std::shared_ptr<ov::Model> model, const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    
    std::vector<ov::Output<ov::Node>> outputs;
    for (const auto& node : nodes) {
        const auto& node_outputs = node->outputs();
        for (auto out : node_outputs)
            outputs.push_back(out);
        
    }
    for (const auto& output : outputs) {
        auto consumers = output.get_target_inputs();
        std::string variable_id = output.get_any_name();
        const auto& var_name = output.get_any_name();
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{output.get_partial_shape(), output.get_element_type(), var_name});
        // Create ReadValue
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(variable);
        for (auto& consumer : consumers) {
            consumer.replace_source_output(read_value->output(0));
        }
        auto assign = std::make_shared<ov::op::v6::Assign>(read_value->output(0), variable);
        model->add_sinks({assign});
    }
    
}

void Graph::convert_stateless_LLM_to_stateful_LLM(std::shared_ptr<ov::Model>& model) {

    std::vector<std::string> past_keys;
    std::vector<std::string> past_values;
    std::vector<std::string> present_keys;
    std::vector<std::string> present_values;
    std::vector<ov::op::v0::Parameter> key_values;
    std::shared_ptr<ov::op::v0::Parameter> beam_idx;
    bool found_input_id = false;
    bool found_input_hidden_states;
    std::string param_name;

    if (model_has_input_output_name(model, "beam_idx")) {
        OPENVINO_ASSERT("Model already has fused cache");
    }
    std::string main_input_name = model_has_input_output_name(model, "input_ids") ? "input_ids" : "input_hidden_states";
    if (main_input_name == "input_ids" || main_input_name == "input_hidden_states")
        found_input_id = true;
    size_t input_id_index = index_of_model_input_output(model, main_input_name);

    PartialShape input_batch_shape = model->input(main_input_name).get_partial_shape();
    Dimension batch_dim = input_batch_shape[0];
    beam_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{batch_dim});
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->output(0).get_tensor().add_names({"beam_idx"});
    const auto& params = model->get_parameters();
    model->add_parameters({beam_idx});
    
    for (auto i = 0; i < params.size(); i++) {
        //iterate over all inputs and make a list of all KV ops
        auto param_name = params.at(i)->output(0).get_any_name();
        auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>({0}));
        if (param_name.find("past_keys") != std::string::npos) {
            past_keys.push_back(param_name);
            if (found_input_id) {
                auto consumers = params.at(i)->output(0).get_target_inputs();
                auto gather_op = std::make_shared<ov::op::v8::Gather>(params.at(i), beam_idx, axis);
                for (const auto& consumer : consumers)
                    consumer.replace_source_output(gather_op);
            }
        }
        if (param_name.find("past_values") != std::string::npos) {
            past_values.push_back(param_name);
            if (found_input_id) {
                auto consumers = params.at(i)->output(0).get_target_inputs();
                auto gather_op = std::make_shared<ov::op::v8::Gather>(params.at(i), beam_idx, axis);
                for (const auto& consumer : consumers)
                    consumer.replace_source_output(gather_op);
            }
        }
        if (param_name.find("past_key_values") != std::string::npos && param_name.find(".key") != std::string::npos) {
            past_keys.push_back(param_name);
            if (found_input_id) {
                auto consumers = params.at(i)->output(0).get_target_inputs();
                auto gather_op = std::make_shared<ov::op::v8::Gather>(params.at(i), beam_idx, axis);
                for (const auto& consumer : consumers)
                    consumer.replace_source_output(gather_op);
            }
        }
        if (param_name.find("past_key_values") != std::string::npos && param_name.find(".value") != std::string::npos) {
            past_values.push_back(param_name);
            if (found_input_id) {
                auto consumers = params.at(i)->output(0).get_target_inputs();
                auto gather_op = std::make_shared<ov::op::v8::Gather>(params.at(i), beam_idx, axis);
                for (const auto& consumer : consumers)
                    consumer.replace_source_output(gather_op);
            }
        }
            
    }
    const auto& results = model->get_results();
    for(size_t i = 0; i < results.size(); i++){

        auto res_name = results.at(i)->output(0).get_any_name();
        if (res_name.find("present_keys") != std::string::npos)
            present_keys.push_back(res_name);
        if (res_name.find("present_values") != std::string::npos)
             present_values.push_back(res_name);
        if (res_name.find("present") != std::string::npos && res_name.find(".key") != std::string::npos)
            present_keys.push_back(res_name);
        if (res_name.find("present") != std::string::npos && res_name.find(".value") != std::string::npos)
            present_values.push_back(res_name);

    }
    std::map<std::string, std::string> param_res_names;
    OPENVINO_ASSERT(past_values.size() == past_keys.size());
    OPENVINO_ASSERT(present_keys.size() == present_values.size());
    for (size_t i = 0; i < past_keys.size(); i++) {

        param_res_names.insert({past_keys[i], present_keys[i]});
        param_res_names.insert({past_values[i], present_values[i]});

    }
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MakeStateful>(param_res_names);
    manager.run_passes(model);

    
    if (found_input_id) {
        auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(params.at(input_id_index));
        auto axis = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({0}));
        auto index = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({0}));
        auto gather_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>({0}));
        auto batch = std::make_shared<ov::op::v8::Gather>(shapeOf, index, gather_axis);
        std::vector<int> dims;
        auto batch_dim = 0;
        for (auto op : model->get_ops()) {
            std::string str1 = "ReadValue";
            if (str1.compare(op->get_type_name()) == 0) {
                std::vector<std::shared_ptr<ov::Node>> processed_dims;
                processed_dims.push_back(batch);
                auto output_shape = op->get_output_partial_shape(0);
                
                for (auto i = 0; i < output_shape.size(); i++) {
                    dims.push_back(output_shape[i].get_min_length());                    
                }
               
                for (auto i = 1; i < dims.size(); i++) {
                    if (dims[i] != 0) {
                        auto dim_tensor = ov::op::v0::Constant::create(ov::element::i64, {1}, {dims[i]});
                        processed_dims.push_back(dim_tensor);
                    } else {
                        auto dim_tensor = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                        processed_dims.push_back(dim_tensor);  // Placeholder for dynamic dimension
                    }
                }               
                auto shape = std::make_shared<ov::op::v0::Concat>(processed_dims, 0);
                const auto broadcast_const = ov::op::v0::Constant::create(op->get_output_element_type(0), Shape{}, {0});
                auto broadcast = std::make_shared<ov::op::v3::Broadcast>(broadcast_const, shape);
                op->set_arguments(OutputVector{broadcast});
                processed_dims.clear();
                dims.clear();
            }
            
        }
        //model->validate_nodes_and_infer_types();
    }
    if (model_has_input_output_name(model, "encoder_hidden_states")) {
        auto encoder_input = model->input("encoder_hidden_states").get_node_shared_ptr();
        // Find dependent output nodes of the subgraph
        std::unordered_set<std::shared_ptr<ov::Node>> sources = {encoder_input};
        auto output_nodes = find_output_nodes_of_dependent_subgraph(model, sources);
        // Insert ReadValue/Assign stateful ops for those outputs
        insert_state_for_nodes(model, output_nodes);
    }
}

void Graph::remove_dangling_parameters() {
    const auto any_tensor_name_matches_onnx_output = [](const Output<ov::Node>& param_output, const GraphProto& graph) {
        const auto found_in_outputs = [&graph](const std::string& tensor_name) {
            const auto& graph_outputs = graph.output();
            return std::any_of(std::begin(graph_outputs),
                               std::end(graph_outputs),
                               [&tensor_name](const ValueInfoProto& output) {
                                   return tensor_name == output.name();
                               });
        };
        const auto& param_tensor_names = param_output.get_tensor().get_names();
        return std::any_of(std::begin(param_tensor_names), std::end(param_tensor_names), found_in_outputs);
    };

    for (auto param_it = m_parameters.begin(); param_it != m_parameters.end();) {
        auto output = (*param_it)->output(0);
        if (output.get_target_inputs().size() == 0) {
            if (!any_tensor_name_matches_onnx_output(output, m_model->get_graph())) {
                m_cache->remove_node(param_it->get()->get_friendly_name());
                param_it = m_parameters.erase(param_it);
                continue;
            }
        }
        param_it++;
    }
}

void Graph::set_metadata(std::shared_ptr<ov::Model>& model) const {
    const std::string framework_section = "framework";
    const auto metadata = m_model->get_metadata();

    for (const auto& pair : metadata) {
        model->set_rt_info(pair.second, framework_section, pair.first);
    }
}

std::shared_ptr<ov::Model> Graph::convert() {
    convert_to_ov_nodes();
    remove_dangling_parameters();
    auto function = create_model();
    convert_stateless_LLM_to_stateful_LLM(function);
    set_metadata(function);
    return function;
}

ov::OutputVector Graph::make_framework_nodes(const Node& onnx_node) {
    std::shared_ptr<ov::frontend::onnx::ONNXFrameworkNode> framework_node;
    if (onnx_node.has_subgraphs()) {
        const auto& subgraphs = onnx_node.get_subgraphs();
        auto inputs = onnx_node.get_ov_inputs();
        std::vector<std::shared_ptr<ov::Model>> models;
        for (const auto& kv : subgraphs) {
            auto& subgraph = kv.second;
            models.push_back(subgraph->decode());
            for (const auto& input : subgraph->get_inputs_from_parent()) {
                const auto& name = input.get_node()->get_friendly_name();
                if (std::find_if(inputs.begin(), inputs.end(), [&name](const Output<ov::Node>& n) -> bool {
                        return name == n.get_node()->get_friendly_name();
                    }) == inputs.end()) {
                    inputs.push_back(input);
                }
            }
        }
        framework_node = std::make_shared<ov::frontend::onnx::ONNXSubgraphFrameworkNode>(onnx_node, models, inputs);
    } else {
        framework_node = std::make_shared<ov::frontend::onnx::ONNXFrameworkNode>(onnx_node);
    }
    return framework_node->outputs();
}

void Graph::decode_to_framework_nodes() {
    const float total = static_cast<float>(m_model->get_graph().node().size());
    unsigned int completed = 0u;
    std::map<std::string, uint64_t> op_statistics;
    // Process ONNX graph nodes, convert to OV nodes
    for (const auto& node_proto : m_model->get_graph().node()) {
        if (m_extensions.telemetry) {
            std::string op_name =
                (node_proto.has_domain() && node_proto.domain() != ""
                     ? "***." + node_proto.op_type() + "-X"
                     : node_proto.op_type() + +"-" + std::to_string(m_model->get_opset_version(node_proto.domain())));
            op_statistics[op_name]++;
        }
        const Node node{node_proto, this};
        ov::OutputVector ov_nodes{make_framework_nodes(node)};
        set_friendly_names(node, ov_nodes);
        // Iterate over the number of outputs for given node in graph.
        // Some of them may be optional and trimmed. See:
        // https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
        for (std::size_t i{0}; i < node.get_outputs_size(); ++i) {
            m_cache->emplace_node(node.output(static_cast<int>(i)), std::move(ov_nodes.at(i)));
        }
        ++completed;
        m_extensions.progress_reporter->report_progress(completed / total, static_cast<unsigned int>(total), completed);
    }
    if (m_extensions.telemetry) {
        for (const auto& op : op_statistics) {
            m_extensions.telemetry->send_event("op_count", "onnx_" + op.first, static_cast<int>(op.second));
        }
    }
}

std::shared_ptr<ov::Model> Graph::create_model() {
    auto model = std::make_shared<ov::Model>(get_ov_outputs(), m_parameters, get_name());
    const auto& onnx_outputs = m_model->get_graph().output();
    for (std::size_t i{0}; i < model->get_output_size(); ++i) {
        const auto& result_node = model->get_output_op(i);
        const std::string onnx_output_name = onnx_outputs.Get(static_cast<int>(i)).name();
        result_node->set_friendly_name(onnx_output_name + "/sink_port_0");
        const auto& previous_operation = result_node->get_input_node_shared_ptr(0);
        previous_operation->set_friendly_name(onnx_output_name);
    }
    return model;
}

std::shared_ptr<ov::Model> Graph::decode() {
    decode_to_framework_nodes();
    auto model = create_model();
    auto& rt_info = model->get_rt_info();
    rt_info[ONNX_GRAPH_RT_ATTRIBUTE] = shared_from_this();
    return model;
}

bool Graph::is_ov_node_in_cache(const std::string& name) const {
    return m_cache->contains(name);
}

Output<ov::Node> Graph::get_ov_node_from_cache(const std::string& name) {
    return m_cache->get_node(name);
}

ov::OutputVector Graph::get_ov_outputs() {
    ov::OutputVector results;
    for (const auto& output : m_model->get_graph().output()) {
        const auto& ov_output = get_ov_node_from_cache(output.name());
        if (!ov::op::util::is_null(ov_output))  // ignore optional outputs
        {
            results.emplace_back(ov_output);
        }
    }
    return results;
}

ov::OutputVector Graph::make_ov_nodes(const Node& onnx_node) {
    ov::OutputVector ov_subgraph_outputs;
    std::string error_message;
    const std::string onnx_prefix = "[ONNX Frontend] ";
    // Failed to convert operation "
    if (m_model->is_operator_available(onnx_node.op_type(), onnx_node.domain())) {
        const auto ng_node_factory = m_model->get_operator(onnx_node.op_type(), onnx_node.domain());
        try {
            ov_subgraph_outputs = ng_node_factory(onnx_node);
        } catch (const ::ov::frontend::onnx::error::OnnxNodeValidationFailure& e) {
            error_message = e.what();
        } catch (const std::exception& exc) {
            error_message = error::detail::get_error_msg_prefix(onnx_node);
            error_message += ": " + std::string{exc.what()};
        } catch (...) {
            error_message = error::detail::get_error_msg_prefix(onnx_node);
            // Since we do not know anything about current exception data type we can only
            // notify user in this way.
            error_message += "Unhandled exception type. \n";
        }
    }
    if (ov_subgraph_outputs.empty()) {  // translation not possible (not supported op or exception during processing)
        if (m_extensions.telemetry && !error_message.empty()) {
            std::string onnx_domain = onnx_node.domain();
            int64_t opset_version = m_model->get_opset_version(onnx_domain);
            error_message = onnx_prefix + "Conversion failed for " +
                            (onnx_domain != "" ? "***." + onnx_node.op_type() + "-X"
                                               : onnx_node.op_type() + "-" + std::to_string(opset_version)) +
                            "\n" + error_message;
        }
        const auto not_supported_node =
            std::make_shared<ov::frontend::onnx::NotSupportedONNXNode>(onnx_node.get_ov_inputs(),
                                                                       onnx_node.get_outputs_size(),
                                                                       onnx_node.domain(),
                                                                       onnx_node.op_type(),
                                                                       error_message);
        ov_subgraph_outputs = not_supported_node->outputs();
    }

    const size_t outputs_size = std::accumulate(std::begin(ov_subgraph_outputs),
                                                std::end(ov_subgraph_outputs),
                                                static_cast<size_t>(0),
                                                [](const size_t lhs, const Output<ov::Node>& rhs) {
                                                    return lhs + rhs.get_node()->get_output_size();
                                                });
    FRONT_END_GENERAL_CHECK(onnx_node.get_outputs_size() <= outputs_size,
                            "Expected output number of ",
                            onnx_node.op_type(),
                            " node is ",
                            onnx_node.get_outputs_size(),
                            " while the implementation provides ",
                            outputs_size,
                            " outputs");

    set_friendly_names(onnx_node, ov_subgraph_outputs);

    for (std::size_t i{0}; i < onnx_node.get_outputs_size(); ++i) {
        auto ov_node_output = ov_subgraph_outputs.at(i);
        m_cache->emplace_node(onnx_node.output(static_cast<int>(i)), std::move(ov_node_output));
    }

    return ov_subgraph_outputs;
}

void Graph::set_friendly_names(const Node& onnx_node, const ov::OutputVector& ov_subgraph_outputs) const {
    if (std::all_of(std::begin(ov_subgraph_outputs), std::end(ov_subgraph_outputs), common::is_optimized_out)) {
        for (size_t i = 0; i < ov_subgraph_outputs.size(); ++i) {
            ov_subgraph_outputs[i].get_tensor().add_names({onnx_node.output(static_cast<int>(i))});
            ov_subgraph_outputs[i].get_node_shared_ptr()->set_friendly_name(onnx_node.output(static_cast<int>(i)));
        }
        return;
    }

    const auto common_node = detail::common_node_for_all_outputs(ov_subgraph_outputs);

    const auto ov_subgraph_output_size = static_cast<int>(ov_subgraph_outputs.size());
    for (int i = 0; i < ov_subgraph_output_size; ++i) {
        // Trailing optional outputs may not be specified in the ONNX model.
        // Other optional outputs should have name set to an empty string.
        if (i >= static_cast<int>(onnx_node.get_outputs_size())) {
            break;
        }

        const auto& onnx_node_name = onnx_node.get_name();
        if (onnx_node_name.empty()) {
            // for multioutput nodes, their friendly name is always set to the last ONNX output's name
            // this is because this setter is called in a loop and the last call is ultimate for a given node
            ov_subgraph_outputs[i].get_node()->set_friendly_name(onnx_node.output(i));
        } else {
            if (common_node) {
                ov_subgraph_outputs[i].get_node()->set_friendly_name(onnx_node.get_name());
            } else {
                // if different outputs are produced by different nodes, then those nodes need to be given
                // unique friendly names
                ov_subgraph_outputs[i].get_node()->set_friendly_name(onnx_node.get_name() + "_" + onnx_node.output(i));
            }
        }

        // null node does not have tensor
        if (!ov::op::util::is_null(ov_subgraph_outputs[i])) {
            ov_subgraph_outputs[i].get_tensor().set_names({onnx_node.output(static_cast<int>(i))});
        }
    }
}

const OpsetImports& Graph::get_opset_imports() const {
    return m_model->get_opset_imports();
}

Subgraph::Subgraph(const std::shared_ptr<ModelProto>& model_proto, Graph* parent_graph)
    : Graph(parent_graph->model_dir(),
            model_proto,
            common::make_unique<GraphCache>(),
            parent_graph->get_mmap_cache(),
            detail::subgraph_required_extensions(parent_graph->get_extensions())),
      m_parent_graph(parent_graph) {}

bool Subgraph::is_ov_node_in_cache(const std::string& name) const {
    if (m_cache->contains(name)) {
        return true;
    }
    return m_parent_graph->is_ov_node_in_cache(name);
}

Output<ov::Node> Subgraph::get_ov_node_from_cache(const std::string& name) {
    if (m_cache->contains(name)) {
        return m_cache->get_node(name);
    }
    const auto from_parent_node = m_parent_graph->get_ov_node_from_cache(name);
    if (ov::op::util::is_constant(from_parent_node.get_node()))
        return from_parent_node;
    auto new_param = std::make_shared<ov::op::v0::Parameter>(from_parent_node.get_element_type(),
                                                             from_parent_node.get_partial_shape());
    m_parameter_to_parent_node_map.insert({new_param, name});
    m_cache->emplace_node(name, new_param);
    m_parameters.push_back(new_param);
    m_inputs_from_parent.push_back(name);
    return new_param;
}

std::shared_ptr<ov::Model> Subgraph::convert() {
    convert_to_ov_nodes();
    return create_model();
}

const std::vector<Output<ov::Node>> Subgraph::get_inputs_from_parent() const {
    ov::OutputVector result;
    for (const auto& name : m_inputs_from_parent) {
        result.push_back(m_parent_graph->get_ov_node_from_cache(name));
    }
    return result;
}

void Subgraph::infer_inputs_from_parent() {
    for (auto& it : m_parameter_to_parent_node_map) {
        const auto& node = m_parent_graph->get_ov_node_from_cache(it.second);
        auto& parameter = it.first;
        parameter->set_element_type(node.get_element_type());
        parameter->set_partial_shape(node.get_partial_shape());
    }
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
