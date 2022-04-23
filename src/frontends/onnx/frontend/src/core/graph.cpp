// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/graph.hpp"

#include <exception>
#include <functional>
#include <numeric>
#include <sstream>

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
#include "ops_bridge.hpp"
#include "utils/common.hpp"
#include "utils/legacy_conversion_extension.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {
std::string to_string(const std::map<std::string, std::reference_wrapper<const ONNX_NAMESPACE::NodeProto>>& map) {
    std::string result;
    for (auto it = std::begin(map); it != std::end(map); ++it) {
        result += (it != std::begin(map) ? ", " : "") + it->first;
    }
    return result;
}

inline std::string generate_result_name(const std::string& onnx_output_name,
                                        const std::shared_ptr<ov::Node>& result_node) {
    auto output_index = result_node->input(0).get_source_output().get_index();
    return onnx_output_name + "/sink_port_" + std::to_string(output_index);
}

/// \brief      Gets the operator represented by provided node unique identificator.
///
/// \param[in]  node_proto  The node protobuf representation object.
///
/// \note       The operator is uniquely identified by the tuple (domain, op_type,
///             since_version). The first two elements are stored in NodeProto object,
///             thus we use only them.
///
/// \return     The unique identificator.
///
static std::string get_op_domain_and_name(const ONNX_NAMESPACE::NodeProto& node_proto) {
    std::string domain = get_node_domain(node_proto);
    return (domain.empty() ? "" : domain + ".") + node_proto.op_type();
}

OperatorsBridge init_ops_bridge(const std::vector<ov::frontend::ConversionExtensionBase::Ptr>& conversions) {
    OperatorsBridge bridge;
    // TODO - apply custom conversions to the bridge object
    for (const auto& extension : conversions) {
        if (auto common_conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtension>(extension)) {
            // for (int i = 1; i < ngraph::onnx_import::OperatorsBridge::LATEST_SUPPORTED_ONNX_OPSET_VERSION; ++i) {
            //     const auto converter = common_conv_ext->get_converter();
            //     OperatorsBridge::register_operator(
            //         common_conv_ext->get_op_type(),
            //         i,
            //         "",
            //         [converter](const ngraph::onnx_import::Node& context) -> OutputVector {
            //             return converter(ov::frontend::onnx::NodeContext(context));
            //         });
            // }
        } else if (const auto onnx_conv_ext =
                       std::dynamic_pointer_cast<ov::frontend::onnx::ConversionExtension>(extension)) {
            //     for (int i = 1; i < ngraph::onnx_import::OperatorsBridge::LATEST_SUPPORTED_ONNX_OPSET_VERSION; ++i)
            //         ngraph::onnx_import::register_operator(onnx_conv_ext->get_op_type(),
            //                                                i,
            //                                                "",
            //                                                [=](const ngraph::onnx_import::Node& context) ->
            //                                                OutputVector
            //                                                {
            //                                                    return
            //                                                    onnx_conv_ext->get_converter()(NodeContext(context));
            //                                                });
        } else if (const auto legacy_conv_extension = std::dynamic_pointer_cast<LegacyConversionExtension>(extension)) {
            return legacy_conv_extension->ops_bridge();
        }
    }
    return bridge;
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
}  // namespace detail

Graph::Graph(const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model_proto, ov::frontend::ExtensionHolder extensions)
    : Graph(model_proto, common::make_unique<GraphCache>(), std::move(extensions)) {}

Graph::Graph(const std::shared_ptr<ONNX_NAMESPACE::ModelProto>& model_proto,
             std::unique_ptr<GraphCache>&& cache,
             ov::frontend::ExtensionHolder extensions)
    : m_cache{std::move(cache)},
      m_extensions{std::move(extensions)} {
    const auto ops_bridge = detail::init_ops_bridge(m_extensions.conversions);
    m_model = common::make_unique<Model>(model_proto, detail::build_model_opset(*model_proto, ops_bridge));

    std::map<std::string, Tensor> initializers;

    // Process all initializers in the graph
    for (const auto& initializer_tensor : m_model->get_graph().initializer()) {
        if (initializer_tensor.has_name()) {
            Tensor tensor = Tensor{initializer_tensor};
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

    // Verify that ONNX graph contains only nodes of available operator types
    std::map<std::string, std::reference_wrapper<const ONNX_NAMESPACE::NodeProto>> unknown_operators;
    std::map<std::string, uint64_t> op_statistics;
    for (const auto& node_proto : m_model->get_graph().node()) {
        if (m_extensions.telemetry) {
            op_statistics[node_proto.op_type()]++;
        }
        if (!m_model->is_operator_available(node_proto)) {
            unknown_operators.emplace(detail::get_op_domain_and_name(node_proto), node_proto);
            // If a node from an unregistered domain is detected, try registering that domain
            m_model->enable_opset_domain(get_node_domain(node_proto), ops_bridge);
        }
    }

    if (m_extensions.telemetry) {
        for (const auto& op : op_statistics) {
            m_extensions.telemetry->send_event("op_count", "onnx_" + op.first, op.second);
        }
    }

    // Reverify wheter we still have any unavailable operators.
    auto it = std::begin(unknown_operators);
    while (it != std::end(unknown_operators)) {
        if (m_model->is_operator_available(it->second)) {
            it = unknown_operators.erase(it);
        } else {
            it++;
        }
    }

    NGRAPH_CHECK(unknown_operators.empty(),
                 "OpenVINO does not support the following ONNX operations: ",
                 detail::to_string(unknown_operators));
}

void Graph::convert_to_ngraph_nodes() {
    const float total = static_cast<float>(m_model->get_graph().node().size());
    unsigned int completed = 0u;
    // Process ONNX graph nodes, convert to nGraph nodes
    for (const auto& node_proto : m_model->get_graph().node()) {
        const Node node{node_proto, *this};
        if (node.has_subgraphs()) {
            const auto& subgraphs = node.get_subgraphs();
            for (auto& kv : subgraphs) {
                auto& subgraph = kv.second;
                subgraph->convert();
            }
        }
        OutputVector ng_nodes{make_ng_nodes(node)};
        ++completed;
        m_extensions.progress_reporter->report_progress(completed / total, total, completed);
    }
}

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

std::shared_ptr<Function> Graph::convert() {
    convert_to_ngraph_nodes();
    remove_dangling_parameters();
    return create_function();
}

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
    // Process ONNX graph nodes, convert to nGraph nodes
    for (const auto& node_proto : m_model->get_graph().node()) {
        const Node node{node_proto, *this};
        OutputVector ng_nodes{make_framework_nodes(node)};
        set_friendly_names(node, ng_nodes);
        // Iterate over the number of outputs for given node in graph.
        // Some of them may be optional and trimmed. See:
        // https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
        for (std::size_t i{0}; i < node.get_outputs_size(); ++i) {
            m_cache->emplace_node(node.output(i), std::move(ng_nodes.at(i)));
        }
        ++completed;
        m_extensions.progress_reporter->report_progress(completed / total, total, completed);
    }
}

std::shared_ptr<Function> Graph::create_function() {
    auto function = std::make_shared<Function>(get_ng_outputs(), m_parameters, get_name());
    const auto& onnx_outputs = m_model->get_graph().output();
    for (std::size_t i{0}; i < function->get_output_size(); ++i) {
        // the suffix makes the Result's name unique in case the nodes in the model don't have a name
        auto ov_result = function->get_output_op(i);
        ov_result->set_friendly_name(detail::generate_result_name(onnx_outputs.Get(i).name(), ov_result));
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

Output<ngraph::Node> Graph::get_ng_node_from_cache(const std::string& name) const {
    return m_cache->get_node(name);
}

OutputVector Graph::get_ng_outputs() const {
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
    const auto ng_node_factory = m_model->get_operator(onnx_node.op_type(), onnx_node.domain());
    // contains outputs of nG subgraph implementing a particular ONNX node (possibly a single output of a single node)
    OutputVector ng_subgraph_outputs;
    try {
        ng_subgraph_outputs = ng_node_factory(onnx_node);
    } catch (const ::ngraph::onnx_import::error::OnnxNodeValidationFailure&) {
        // Do nothing OnnxNodeValidationFailure exception already has ONNX node information.
        throw;
    } catch (const std::exception& exc) {
        std::string msg_prefix = error::detail::get_error_msg_prefix(onnx_node);
        throw ngraph_error(msg_prefix + ":\n" + std::string(exc.what()));
    } catch (...) {
        std::string msg_prefix = error::detail::get_error_msg_prefix(onnx_node);
        // Since we do not know anything about current exception data type we can only
        // notify user in this way.
        NGRAPH_ERR << msg_prefix + "Unhandled exception type. \n";
        std::rethrow_exception(std::current_exception());
    }

    set_friendly_names(onnx_node, ng_subgraph_outputs);

    for (std::size_t i{0}; i < onnx_node.get_outputs_size(); ++i) {
        auto ng_node_output = ng_subgraph_outputs.at(i);
        m_cache->emplace_node(onnx_node.output(i), std::move(ng_node_output));
    }

    return ng_subgraph_outputs;
}

void Graph::set_friendly_names(const Node& onnx_node, const OutputVector& ng_subgraph_outputs) const {
    if (onnx_node.op_type() == "Identity") {
        for (size_t i = 0; i < ng_subgraph_outputs.size(); ++i) {
            ng_subgraph_outputs[i].get_tensor().add_names({onnx_node.output(i)});
            ng_subgraph_outputs[i].get_node_shared_ptr()->set_friendly_name(onnx_node.output(i));
        }
        return;
    }

    for (size_t i = 0; i < ng_subgraph_outputs.size(); ++i) {
        // Trailing optional outputs may not be specified in the ONNX model.
        // Other optional outputs should have name set to an empty string.
        if (i >= onnx_node.get_outputs_size()) {
            break;
        }

        ng_subgraph_outputs[i].get_node()->set_friendly_name(onnx_node.output(i));

        // null node does not have tensor
        if (!ngraph::op::is_null(ng_subgraph_outputs[i])) {
            ng_subgraph_outputs[i].get_tensor().set_names({onnx_node.output(i)});
            NGRAPH_SUPPRESS_DEPRECATED_START
            ng_subgraph_outputs[i].get_tensor().set_name(onnx_node.output(i));
            NGRAPH_SUPPRESS_DEPRECATED_END
        }
    }
}

const OpsetImports& Graph::get_opset_imports() const {
    return m_model->get_opset_imports();
}

Subgraph::Subgraph(std::shared_ptr<ONNX_NAMESPACE::ModelProto> model_proto, const Graph* parent_graph)
    : Graph(model_proto, common::make_unique<GraphCache>()),
      m_parent_graph(parent_graph) {
    // do not copy a pre-configured progress reporter extension to the subgraph
    m_extensions.telemetry = parent_graph->get_extensions().telemetry;
    m_extensions.conversions = parent_graph->get_extensions().conversions;
}

bool Subgraph::is_ng_node_in_cache(const std::string& name) const {
    if (m_cache->contains(name)) {
        return true;
    }
    return m_parent_graph->is_ng_node_in_cache(name);
}

Output<ngraph::Node> Subgraph::get_ng_node_from_cache(const std::string& name) const {
    if (m_cache->contains(name)) {
        return m_cache->get_node(name);
    }
    return m_parent_graph->get_ng_node_from_cache(name);
}

OutputVector Subgraph::make_ng_nodes(const Node& onnx_node) {
    replace_input_from_parent_scope_with_parameter(onnx_node);
    return Graph::make_ng_nodes(onnx_node);
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

OutputVector Subgraph::make_framework_nodes(const Node& onnx_node) {
    replace_input_from_parent_scope_with_parameter(onnx_node);
    return Graph::make_framework_nodes(onnx_node);
}

void Subgraph::replace_input_from_parent_scope_with_parameter(const Node& onnx_node) {
    for (std::size_t i = 0; i < onnx_node.get_inputs_size(); ++i) {
        const auto& in_name = onnx_node.input(i);
        if (m_parent_graph->is_ng_node_in_cache(in_name) &&
            std::find(m_inputs_from_parent.begin(), m_inputs_from_parent.end(), in_name) ==
                m_inputs_from_parent.end()) {
            const auto& from_parent_node = m_parent_graph->get_ng_node_from_cache(in_name);
            if (op::is_constant(from_parent_node.get_node()))
                continue;
            auto new_param = std::make_shared<ngraph::op::Parameter>(from_parent_node.get_element_type(),
                                                                     from_parent_node.get_partial_shape());
            m_parameter_to_parent_node_map.insert({new_param, in_name});
            m_cache->emplace_node(in_name, new_param);
            m_parameters.push_back(new_param);
            m_inputs_from_parent.push_back(in_name);
        }
    }
}

}  // namespace onnx_import

}  // namespace ngraph
