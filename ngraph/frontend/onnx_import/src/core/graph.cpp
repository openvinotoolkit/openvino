// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exception>
#include <functional>
#include <numeric>
#include <sstream>

#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "exceptions.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/provenance.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/common.hpp"
#include "utils/provenance_tag.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            static std::string to_string(
                const std::map<std::string,
                               std::reference_wrapper<const ONNX_NAMESPACE::NodeProto>>& map)
            {
                std::string result;
                for (auto it = std::begin(map); it != std::end(map); ++it)
                {
                    result += (it != std::begin(map) ? ", " : "") + it->first;
                }
                return result;
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
            static std::string get_op_domain_and_name(const ONNX_NAMESPACE::NodeProto& node_proto)
            {
                std::string domain = get_node_domain(node_proto);
                return (domain.empty() ? "" : domain + ".") + node_proto.op_type();
            }
        } // namespace detail

        Graph::Graph(std::unique_ptr<Model>&& model)
            : Graph(std::move(model), common::make_unique<GraphCache>())
        {
            // Remove dangling Parameters
            for (auto param_it = m_parameters.begin(); param_it != m_parameters.end();)
            {
                if ((*param_it)->get_output_target_inputs(0).size() == 0)
                {
                    const auto& name = (*param_it)->get_friendly_name();
                    auto out_it = std::find_if(
                        m_outputs.begin(), m_outputs.end(), [&name](const ValueInfo& info) {
                            return info.get_name() == name;
                        });
                    if (out_it == m_outputs.end())
                    {
                        m_cache->remove_node(name);
                        param_it = m_parameters.erase(param_it);
                        continue;
                    }
                }
                param_it++;
            }
        }

        Graph::Graph(std::unique_ptr<Model>&& model, std::unique_ptr<GraphCache>&& cache)
            : m_model{std::move(model)}
            , m_cache{std::move(cache)}
        {
            std::map<std::string, Tensor> initializers;
            // Process all initializers in the graph
            for (const auto& initializer_tensor : m_model->get_graph().initializer())
            {
                if (initializer_tensor.has_name())
                {
                    Tensor tensor = Tensor{initializer_tensor};
                    std::shared_ptr<default_opset::Constant> ng_constant;
                    // For each initializer create a Constant node and store it in cache
                    try
                    {
                        ng_constant = tensor.get_ng_constant();
                    }
                    catch (const error::invalid_external_data&)
                    {
                        // invalid external data makes initializers creation impossible
                        throw;
                    }
                    catch (const ngraph::ngraph_error& exc)
                    {
                        NGRAPH_WARN
                            << "\nCould not create an nGraph Constant for initializer '"
                            << initializer_tensor.name() << "'. \n"
                            << "Constant with a 0 value was created, make sure connected input is "
                               "optional.\n"
                            << "Otherwise verify if the initializer contains a correct number of "
                               "elements matching the initializer's shape. \n"
                            << "Detailed error:\n"
                            << exc.what();
                        ng_constant =
                            default_opset::Constant::create(tensor.get_ng_type(), Shape{}, {0});
                    }

                    initializers.emplace(initializer_tensor.name(), tensor);
                    add_provenance_tag_to_initializer(tensor, ng_constant);
                    m_cache->emplace_node(initializer_tensor.name(), std::move(ng_constant));
                }
            }

            // Process all ONNX graph inputs, convert them to nGraph nodes and store in cache
            for (const auto& input : m_model->get_graph().input())
            {
                m_inputs.emplace_back(input);

                // Check if a Constant node was already created from an initializer
                if (m_cache->contains(input.name()))
                {
                    continue;
                }

                const auto value_info = m_inputs.back();
                auto ng_node = value_info.get_ng_node(m_parameters, initializers);
                add_provenance_tag_to_input(value_info, ng_node);
                m_cache->emplace_node(input.name(), std::move(ng_node));
            }

            // Process all graph outputs
            for (const auto& output : m_model->get_graph().output())
            {
                m_outputs.emplace_back(output);
            }

            // Verify that ONNX graph contains only nodes of available operator types
            std::map<std::string, std::reference_wrapper<const ONNX_NAMESPACE::NodeProto>>
                unknown_operators;
            for (const auto& node_proto : m_model->get_graph().node())
            {
                if (!m_model->is_operator_available(node_proto))
                {
                    unknown_operators.emplace(detail::get_op_domain_and_name(node_proto),
                                              node_proto);
                    // If a node from an unregistered domain is detected, try registering that
                    // domain
                    m_model->enable_opset_domain(get_node_domain(node_proto));
                }
            }

            // Reverify wheter we still have any unavailable operators.
            auto it = std::begin(unknown_operators);
            while (it != std::end(unknown_operators))
            {
                if (m_model->is_operator_available(it->second))
                {
                    it = unknown_operators.erase(it);
                }
                else
                {
                    it++;
                }
            }

            NGRAPH_CHECK(unknown_operators.empty(),
                         "nGraph does not support the following ONNX operations: ",
                         detail::to_string(unknown_operators));

            // Process ONNX graph nodes, convert to nGraph nodes
            for (const auto& node_proto : m_model->get_graph().node())
            {
                m_nodes.emplace_back(node_proto, *this);
                const Node& node{m_nodes.back()};

                OutputVector ng_nodes{node.get_ng_nodes()};
                // Iterate over the number of outputs for given node in graph.
                // Some of them may be optional and trimmed. See:
                // https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
                for (std::size_t i{0}; i < node.get_outputs_size(); ++i)
                {
                    m_cache->emplace_node(node.output(i), std::move(ng_nodes.at(i)));
                }
            }
        }

        const GraphCache& Graph::get_graph_cache() const { return *m_cache.get(); }
        bool Graph::is_node_in_cache(const std::string& name) const
        {
            return m_cache->contains(name);
        }

        Output<ngraph::Node> Graph::get_ng_node_from_cache(const std::string& name) const
        {
            return m_cache->get_node(name);
        }

        OutputVector Graph::get_ng_outputs() const
        {
            OutputVector results;
            for (const auto& output : m_model->get_graph().output())
            {
                const auto& ng_output = get_ng_node_from_cache(output.name());
                if (!ngraph::op::is_null(ng_output)) // ignore optional outputs
                {
                    results.emplace_back(ng_output);
                }
            }
            return results;
        }

        OutputVector Graph::make_ng_nodes(const Node& onnx_node) const
        {
            const auto ng_node_factory =
                m_model->get_operator(onnx_node.op_type(), onnx_node.domain());
            OutputVector ng_node_vector;
            try
            {
                ng_node_vector = ng_node_factory(onnx_node);
            }
            catch (const ::ngraph::onnx_import::error::OnnxNodeValidationFailure& exc)
            {
                // Do nothing OnnxNodeValidationFailure exception already has ONNX node information.
                throw;
            }
            catch (const std::exception& exc)
            {
                std::string msg_prefix = error::detail::get_error_msg_prefix(onnx_node);
                throw ngraph_error(msg_prefix + ":\n" + std::string(exc.what()));
            }
            catch (...)
            {
                std::string msg_prefix = error::detail::get_error_msg_prefix(onnx_node);
                // Since we do not know anything about current exception data type we can only
                // notify user in this way.
                NGRAPH_ERR << msg_prefix + "Unhandled exception type. \n";
                std::rethrow_exception(std::current_exception());
            }
            set_friendly_names(onnx_node, ng_node_vector);
            add_provenance_tags(onnx_node, ng_node_vector);

            return ng_node_vector;
        }

        void Graph::set_friendly_names(const Node& onnx_node,
                                       const OutputVector& ng_node_vector) const
        {
            for (size_t i = 0; i < ng_node_vector.size(); ++i)
            {
                // Trailing optional outputs may not be specified in the ONNX model.
                // Other optional outputs should have name set to an empty string.
                if (i >= onnx_node.get_outputs_size())
                {
                    break;
                }

                ng_node_vector[i].get_node()->set_friendly_name(onnx_node.output(i));

                // null node does not have tensor
                if (!ngraph::op::is_null(ng_node_vector[i]))
                {
                    ng_node_vector[i].get_tensor().set_names({onnx_node.output(i)});
                }
            }
        }

        void Graph::add_provenance_tag_to_initializer(
            const Tensor& tensor, std::shared_ptr<default_opset::Constant> node) const
        {
            if (!ngraph::get_provenance_enabled())
            {
                return;
            }

            const std::string tag =
                detail::build_input_provenance_tag(tensor.get_name(), tensor.get_shape());

            node->add_provenance_tag(tag);
        }

        void Graph::add_provenance_tag_to_input(const ValueInfo& input,
                                                std::shared_ptr<ngraph::Node> node) const
        {
            if (!ngraph::get_provenance_enabled())
            {
                return;
            }

            const std::string tag =
                detail::build_input_provenance_tag(input.get_name(), input.get_shape());

            node->add_provenance_tag(tag);
        }

        void Graph::add_provenance_tags(const Node& onnx_node,
                                        const OutputVector& ng_node_vector) const
        {
            if (!ngraph::get_provenance_enabled())
            {
                return;
            }

            const auto tag = detail::build_op_provenance_tag(onnx_node);
            const auto ng_inputs = onnx_node.get_ng_inputs();

            ngraph::traverse_nodes(
                as_node_vector(ng_node_vector),
                [&tag](std::shared_ptr<ngraph::Node> ng_node) { ng_node->add_provenance_tag(tag); },
                as_node_vector(ng_inputs));
        }

        const OpsetImports& Graph::get_opset_imports() const
        {
            return m_model->get_opset_imports();
        }

        Subgraph::Subgraph(std::unique_ptr<Model>&& model, const Graph& parent_graph)
            : Graph(
                  std::move(model),
                  std::unique_ptr<SubgraphCache>(new SubgraphCache(parent_graph.get_graph_cache())))
        {
            // find all nodes on edge parent graph-subgraph
            // (it means input of node from parent graph, output from subgraph)
            for (const auto& node_proto : m_model->get_graph().node())
            {
                int input_index = 0;
                for (const auto& in_name : node_proto.input())
                {
                    if (m_cache->node_scope(in_name) == NodeScope::ParentGraph)
                    {
                        const auto& from_parent_node = m_cache->get_node(in_name);
                        // constants are skipped
                        if (!ngraph::is_type<ngraph::op::Constant>(
                                from_parent_node.get_node_shared_ptr()))
                        {
                            for (const auto& out_name : node_proto.output())
                            {
                                if (m_cache->node_scope(out_name) == NodeScope::SubGraph)
                                {
                                    auto out_node_to_replace_input = m_cache->get_node(out_name);
                                    auto new_param = std::make_shared<ngraph::op::Parameter>(
                                        from_parent_node.get_element_type(),
                                        from_parent_node.get_partial_shape());
                                    // replace input from parent scope with parameter
                                    out_node_to_replace_input.get_node()
                                        ->input(input_index)
                                        .replace_source_output(new_param);
                                    m_parameters.push_back(new_param);
                                    m_outputs_from_parent.push_back(from_parent_node);
                                }
                            }
                        }
                    }
                    ++input_index;
                }
            }
        }

        const std::vector<Output<ngraph::Node>> Subgraph::get_outputs_from_parent() const
        {
            return m_outputs_from_parent;
        }

    } // namespace onnx_import

} // namespace ngraph
