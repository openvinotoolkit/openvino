// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/graph.hpp"

#include <exception>
#include <functional>
#include <numeric>
#include <sstream>

#include "core/transform.hpp"
#include "core/value_info.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "onnx_framework_node.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/frontend/onnx/extension/conversion.hpp"
#include "openvino/frontend/onnx/node_context.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/log.hpp"
#include "utils/common.hpp"
#include "utils/legacy_conversion_extension.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {
bool common_node_for_all_outputs(const OutputVector& outputs) {
    const auto first_out_node = outputs.at(0).get_node();
    bool ret = std::all_of(std::next(std::begin(outputs)),
                           std::end(outputs),
                           [first_out_node](const OutputVector::value_type& output) {
                               return output.get_node() == first_out_node;
                           });
    return ret;
};

OPENVINO_SUPPRESS_DEPRECATED_START
OperatorsBridge register_extensions(OperatorsBridge& bridge,
                                    const std::vector<ov::frontend::ConversionExtensionBase::Ptr>& conversions) {
    for (const auto& extension : conversions) {
        if (const auto common_conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtension>(extension)) {
            bridge.overwrite_operator(
                common_conv_ext->get_op_type(),
                "",
                [common_conv_ext](const ngraph::onnx_import::Node& node) -> OutputVector {
                    return common_conv_ext->get_converter()(ov::frontend::onnx::NodeContext(node));
                });
        } else if (const auto onnx_conv_ext =
                       std::dynamic_pointer_cast<ov::frontend::onnx::ConversionExtension>(extension)) {
            bridge.overwrite_operator(onnx_conv_ext->get_op_type(),
                                      onnx_conv_ext->get_domain(),
                                      [onnx_conv_ext](const ngraph::onnx_import::Node& node) -> OutputVector {
                                          return onnx_conv_ext->get_converter()(ov::frontend::onnx::NodeContext(node));
                                      });
        }
    }
    return bridge;
}
OPENVINO_SUPPRESS_DEPRECATED_END

OperatorsBridge init_ops_bridge(const std::vector<ov::frontend::ConversionExtensionBase::Ptr>& conversions) {
    const auto legacy_conv_ext = std::find_if(std::begin(conversions),
                                              std::end(conversions),
                                              [](const ov::frontend::ConversionExtensionBase::Ptr& conv) {
                                                  return std::dynamic_pointer_cast<LegacyConversionExtension>(conv);
                                              });
    if (legacy_conv_ext == std::end(conversions)) {  // no legacy extensions used
        OperatorsBridge bridge;
        return register_extensions(bridge, conversions);
    } else {  // legacy extensions can be mixed with the new one
        return register_extensions(std::dynamic_pointer_cast<LegacyConversionExtension>(*legacy_conv_ext)->ops_bridge(),
                                   conversions);
    }
}

Model::ModelOpSet build_model_opset(const ONNX_NAMESPACE::ModelProto& model_proto, const OperatorsBridge& ops_bridge) {
    // copy the opset imports from the ONNX model and sort them by their version in ascending order
    // this will make sure that multiple opset imports for the same domain will cause the largest
    // version to be used for this model, for example:
    // [{domain:"", version:11}, {domain:"", version:1} {domain:"", version:13}] ==> {domain:"", version:13}
    auto opset_imports = model_proto.opset_import();
    const auto sort_by_version_ascending = [](const ONNX_NAMESPACE::OperatorSetIdProto& lhs,
                                              const ONNX_NAMESPACE::OperatorSetIdProto& rhs) {
        return lhs.version() < rhs.version();
    };
    std::sort(std::begin(opset_imports), std::end(opset_imports), sort_by_version_ascending);

    Model::ModelOpSet opset;
    std::for_each(opset_imports.rbegin(),
                  opset_imports.rend(),
                  [&opset, &ops_bridge](const ONNX_NAMESPACE::OperatorSetIdProto& onnx_opset) {
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
             const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
             detail::MappedMemoryHandles mmap_cache,
             ov::frontend::ExtensionHolder extensions)
    : Graph(model_dir, model_proto, common::make_unique<GraphCache>(), mmap_cache, std::move(extensions)) {}

Graph::Graph(const std::string& model_dir,
             const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
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
            std::shared_ptr<default_opset::Constant> ng_constant;
            // For each initializer create a Constant node and store it in cache
            try {
                ng_constant = tensor.get_ng_constant();
            } catch (const error::invalid_external_data&) {
                // invalid external data makes initializers creation impossible
                throw;
            } catch (const ngraph::ngraph_error&) {
                ng_constant = ngraph::onnx_import::common::make_failsafe_constant(tensor.get_ng_type());
            }

            initializers.emplace(initializer_tensor.name(), tensor);
            ng_constant->get_output_tensor(0).set_names({initializer_tensor.name()});
            m_cache->emplace_node(initializer_tensor.name(), std::move(ng_constant));
        }
    }

    // Process all ONNX graph inputs, convert them to nGraph nodes and store in cache
    for (const auto& input : m_model->get_graph().input()) {
        // Check if a Constant node was already created from an initializer
        if (m_cache->contains(input.name())) {
            continue;
        }

        ValueInfo value_info{input};
        auto ng_node = value_info.get_ng_node(m_parameters, initializers);
        m_cache->emplace_node(input.name(), std::move(ng_node));
    }
}

OPENVINO_SUPPRESS_DEPRECATED_START
void Graph::convert_to_ngraph_nodes() {
    const float total = static_cast<float>(m_model->get_graph().node().size());
    unsigned int completed = 0u;
    std::map<std::string, uint64_t> op_statistics;
    // Process ONNX graph nodes, convert to nGraph nodes
    for (const auto& node_proto : m_model->get_graph().node()) {
        if (m_extensions.telemetry) {
            op_statistics[node_proto.op_type()]++;
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
        OutputVector ng_nodes{make_ng_nodes(node)};
        ++completed;
        m_extensions.progress_reporter->report_progress(completed / total, static_cast<unsigned int>(total), completed);
    }
    if (m_extensions.telemetry) {
        for (const auto& op : op_statistics) {
            m_extensions.telemetry->send_event("op_count", "onnx_" + op.first, static_cast<int>(op.second));
        }
    }
}
OPENVINO_SUPPRESS_DEPRECATED_END

void Graph::remove_dangling_parameters() {
    const auto any_tensor_name_matches_onnx_output = [](const Output<ov::Node>& param_output,
                                                        const ONNX_NAMESPACE::GraphProto& graph) {
        const auto found_in_outputs = [&graph](const std::string& tensor_name) {
            const auto& graph_outputs = graph.output();
            return std::any_of(std::begin(graph_outputs),
                               std::end(graph_outputs),
                               [&tensor_name](const ONNX_NAMESPACE::ValueInfoProto& output) {
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

std::shared_ptr<Function> Graph::convert() {
    convert_to_ngraph_nodes();
    remove_dangling_parameters();
    auto function = create_function();
    set_metadata(function);
    return function;
}

OPENVINO_SUPPRESS_DEPRECATED_START
OutputVector Graph::make_framework_nodes(const Node& onnx_node) {
    std::shared_ptr<frontend::ONNXFrameworkNode> framework_node;
    if (onnx_node.has_subgraphs()) {
        const auto& subgraphs = onnx_node.get_subgraphs();
        auto inputs = onnx_node.get_ng_inputs();
        std::vector<std::shared_ptr<Function>> functions;
        for (const auto& kv : subgraphs) {
            auto& subgraph = kv.second;
            functions.push_back(subgraph->decode());
            for (const auto& input : subgraph->get_inputs_from_parent()) {
                const auto& name = input.get_node()->get_friendly_name();
                if (std::find_if(inputs.begin(), inputs.end(), [&name](const Output<ngraph::Node>& n) -> bool {
                        return name == n.get_node()->get_friendly_name();
                    }) == inputs.end()) {
                    inputs.push_back(input);
                }
            }
        }
        framework_node = std::make_shared<frontend::ONNXSubgraphFrameworkNode>(onnx_node, functions, inputs);
    } else {
        framework_node = std::make_shared<frontend::ONNXFrameworkNode>(onnx_node);
    }
    return framework_node->outputs();
}

void Graph::decode_to_framework_nodes() {
    const float total = static_cast<float>(m_model->get_graph().node().size());
    unsigned int completed = 0u;
    std::map<std::string, uint64_t> op_statistics;
    // Process ONNX graph nodes, convert to nGraph nodes
    for (const auto& node_proto : m_model->get_graph().node()) {
        if (m_extensions.telemetry) {
            op_statistics[node_proto.op_type()]++;
        }
        const Node node{node_proto, this};
        OutputVector ng_nodes{make_framework_nodes(node)};
        set_friendly_names(node, ng_nodes);
        // Iterate over the number of outputs for given node in graph.
        // Some of them may be optional and trimmed. See:
        // https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
        for (std::size_t i{0}; i < node.get_outputs_size(); ++i) {
            m_cache->emplace_node(node.output(static_cast<int>(i)), std::move(ng_nodes.at(i)));
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
OPENVINO_SUPPRESS_DEPRECATED_END

std::shared_ptr<Function> Graph::create_function() {
    auto function = std::make_shared<Function>(get_ng_outputs(), m_parameters, get_name());
    const auto& onnx_outputs = m_model->get_graph().output();
    for (std::size_t i{0}; i < function->get_output_size(); ++i) {
        const auto& result_node = function->get_output_op(i);
        const std::string onnx_output_name = onnx_outputs.Get(static_cast<int>(i)).name();
        result_node->set_friendly_name(onnx_output_name + "/sink_port_0");
        const auto& previous_operation = result_node->get_input_node_shared_ptr(0);
        previous_operation->set_friendly_name(onnx_output_name);
    }
    return function;
}

std::shared_ptr<Function> Graph::decode() {
    decode_to_framework_nodes();
    auto function = create_function();
    auto& rt_info = function->get_rt_info();
    rt_info[ONNX_GRAPH_RT_ATTRIBUTE] = shared_from_this();
    return function;
}

bool Graph::is_ng_node_in_cache(const std::string& name) const {
    return m_cache->contains(name);
}

Output<ngraph::Node> Graph::get_ng_node_from_cache(const std::string& name) {
    return m_cache->get_node(name);
}

OPENVINO_SUPPRESS_DEPRECATED_START
OutputVector Graph::get_ng_outputs() {
    OutputVector results;
    for (const auto& output : m_model->get_graph().output()) {
        const auto& ng_output = get_ng_node_from_cache(output.name());
        if (!ngraph::op::is_null(ng_output))  // ignore optional outputs
        {
            results.emplace_back(ng_output);
        }
    }
    return results;
}

OutputVector Graph::make_ng_nodes(const Node& onnx_node) {
    OutputVector ng_subgraph_outputs;
    std::string error_message;
    if (m_model->is_operator_available(onnx_node.op_type(), onnx_node.domain())) {
        const auto ng_node_factory = m_model->get_operator(onnx_node.op_type(), onnx_node.domain());
        try {
            ng_subgraph_outputs = ng_node_factory(onnx_node);
        } catch (const ::ngraph::onnx_import::error::OnnxNodeValidationFailure& e) {
            error_message = e.what();
        } catch (const std::exception& exc) {
            error_message = error::detail::get_error_msg_prefix(onnx_node);
            error_message += ":\n" + std::string{exc.what()};
        } catch (...) {
            error_message = error::detail::get_error_msg_prefix(onnx_node);
            // Since we do not know anything about current exception data type we can only
            // notify user in this way.
            error_message += "Unhandled exception type. \n";
        }
    }
    if (ng_subgraph_outputs.empty()) {  // translation not possible (not supported op or exception during processing)
        const auto not_supported_node = std::make_shared<frontend::NotSupportedONNXNode>(onnx_node.get_ng_inputs(),
                                                                                         onnx_node.get_outputs_size(),
                                                                                         onnx_node.domain(),
                                                                                         onnx_node.op_type(),
                                                                                         error_message);
        ng_subgraph_outputs = not_supported_node->outputs();
    }

    const size_t outputs_size = std::accumulate(std::begin(ng_subgraph_outputs),
                                                std::end(ng_subgraph_outputs),
                                                static_cast<size_t>(0),
                                                [](const size_t lhs, const Output<ov::Node>& rhs) {
                                                    return lhs + rhs.get_node()->get_output_size();
                                                });
    NGRAPH_CHECK(onnx_node.get_outputs_size() <= outputs_size,
                 "Expected output number of ",
                 onnx_node.op_type(),
                 " node is ",
                 onnx_node.get_outputs_size(),
                 " while the implementation provides ",
                 outputs_size,
                 " outputs");

    set_friendly_names(onnx_node, ng_subgraph_outputs);

    for (std::size_t i{0}; i < onnx_node.get_outputs_size(); ++i) {
        auto ng_node_output = ng_subgraph_outputs.at(i);
        m_cache->emplace_node(onnx_node.output(static_cast<int>(i)), std::move(ng_node_output));
    }

    return ng_subgraph_outputs;
}

void Graph::set_friendly_names(const Node& onnx_node, const OutputVector& ng_subgraph_outputs) const {
    if (std::all_of(std::begin(ng_subgraph_outputs), std::end(ng_subgraph_outputs), common::is_optimized_out)) {
        for (size_t i = 0; i < ng_subgraph_outputs.size(); ++i) {
            ng_subgraph_outputs[i].get_tensor().add_names({onnx_node.output(static_cast<int>(i))});
            ng_subgraph_outputs[i].get_node_shared_ptr()->set_friendly_name(onnx_node.output(static_cast<int>(i)));
        }
        return;
    }

    const auto common_node = detail::common_node_for_all_outputs(ng_subgraph_outputs);

    const auto ng_subgraph_output_size = static_cast<int>(ng_subgraph_outputs.size());
    for (int i = 0; i < ng_subgraph_output_size; ++i) {
        // Trailing optional outputs may not be specified in the ONNX model.
        // Other optional outputs should have name set to an empty string.
        if (i >= static_cast<int>(onnx_node.get_outputs_size())) {
            break;
        }

        const auto& onnx_node_name = onnx_node.get_name();
        if (onnx_node_name.empty()) {
            // for multioutput nodes, their friendly name is always set to the last ONNX output's name
            // this is because this setter is called in a loop and the last call is ultimate for a given node
            ng_subgraph_outputs[i].get_node()->set_friendly_name(onnx_node.output(i));
        } else {
            if (common_node) {
                ng_subgraph_outputs[i].get_node()->set_friendly_name(onnx_node.get_name());
            } else {
                // if different outputs are produced by different nodes, then those nodes need to be given
                // unique friendly names
                ng_subgraph_outputs[i].get_node()->set_friendly_name(onnx_node.get_name() + "_" + onnx_node.output(i));
            }
        }

        // null node does not have tensor
        if (!ngraph::op::is_null(ng_subgraph_outputs[i])) {
            ng_subgraph_outputs[i].get_tensor().set_names({onnx_node.output(static_cast<int>(i))});
            ov::descriptor::set_ov_tensor_legacy_name(ng_subgraph_outputs[i].get_tensor(),
                                                      onnx_node.output(static_cast<int>(i)));
        }
    }
}
OPENVINO_SUPPRESS_DEPRECATED_END

const OpsetImports& Graph::get_opset_imports() const {
    return m_model->get_opset_imports();
}

Subgraph::Subgraph(const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model_proto, Graph* parent_graph)
    : Graph(parent_graph->model_dir(),
            model_proto,
            common::make_unique<GraphCache>(),
            parent_graph->get_mmap_cache(),
            detail::subgraph_required_extensions(parent_graph->get_extensions())),
      m_parent_graph(parent_graph) {}

bool Subgraph::is_ng_node_in_cache(const std::string& name) const {
    if (m_cache->contains(name)) {
        return true;
    }
    return m_parent_graph->is_ng_node_in_cache(name);
}

Output<ngraph::Node> Subgraph::get_ng_node_from_cache(const std::string& name) {
    if (m_cache->contains(name)) {
        return m_cache->get_node(name);
    }
    const auto from_parent_node = m_parent_graph->get_ng_node_from_cache(name);
    if (ov::op::util::is_constant(from_parent_node.get_node()))
        return from_parent_node;
    auto new_param = std::make_shared<ngraph::op::Parameter>(from_parent_node.get_element_type(),
                                                             from_parent_node.get_partial_shape());
    m_parameter_to_parent_node_map.insert({new_param, name});
    m_cache->emplace_node(name, new_param);
    m_parameters.push_back(new_param);
    m_inputs_from_parent.push_back(name);
    return new_param;
}

std::shared_ptr<Function> Subgraph::convert() {
    convert_to_ngraph_nodes();
    return create_function();
}

const std::vector<Output<ngraph::Node>> Subgraph::get_inputs_from_parent() const {
    OutputVector result;
    for (const auto& name : m_inputs_from_parent) {
        result.push_back(m_parent_graph->get_ng_node_from_cache(name));
    }
    return result;
}

void Subgraph::infer_inputs_from_parent() {
    for (auto& it : m_parameter_to_parent_node_map) {
        const auto& node = m_parent_graph->get_ng_node_from_cache(it.second);
        auto& parameter = it.first;
        parameter->set_element_type(node.get_element_type());
        parameter->set_partial_shape(node.get_partial_shape());
    }
}

}  // namespace onnx_import

}  // namespace ngraph
