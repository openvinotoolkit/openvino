//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <exception>
#include <functional>
#include <numeric>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/provenance.hpp"
#include "onnx_import/core/graph.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/utils/common.hpp"
#include "onnx_import/utils/provenance_tag.hpp"

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

            static std::string get_node_domain(const ONNX_NAMESPACE::NodeProto& node_proto)
            {
                return (node_proto.domain().empty() ? "" : node_proto.domain());
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

        Graph::Graph(const ONNX_NAMESPACE::GraphProto& graph_proto, Model& model)
            : Graph(graph_proto, model, std::unique_ptr<GraphCache>(new GraphCache()))
        {
        }

        Graph::Graph(const ONNX_NAMESPACE::GraphProto& graph_proto,
                     Model& model,
                     std::unique_ptr<GraphCache>&& cache)
            : m_graph_proto{&graph_proto}
            , m_model{&model}
            , m_cache{std::move(cache)}
        {
            std::map<std::string, Tensor> initializers;
            // Process all initializers in the graph
            for (const auto& initializer_tensor : m_graph_proto->initializer())
            {
                if (initializer_tensor.has_name())
                {
                    Tensor tensor = Tensor{initializer_tensor};
                    initializers.emplace(initializer_tensor.name(), tensor);

                    // For each initializer create a Constant node and store it in cache
                    auto ng_constant = tensor.get_ng_constant();
                    add_provenance_tag_to_initializer(tensor, ng_constant);
                    m_cache->emplace_node(initializer_tensor.name(), std::move(ng_constant));
                }
            }

            // Process all ONNX graph inputs, convert them to nGraph nodes and store in cache
            for (const auto& input : m_graph_proto->input())
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
            for (const auto& output : m_graph_proto->output())
            {
                m_outputs.emplace_back(output);
            }

            // Verify that ONNX graph contains only nodes of available operator types
            std::map<std::string, std::reference_wrapper<const ONNX_NAMESPACE::NodeProto>>
                unknown_operators;
            for (const auto& node_proto : m_graph_proto->node())
            {
                if (!m_model->is_operator_available(node_proto))
                {
                    unknown_operators.emplace(detail::get_op_domain_and_name(node_proto),
                                              node_proto);
                    // If a node from an unregistered domain is detected, try registering that
                    // domain
                    m_model->enable_opset_domain(detail::get_node_domain(node_proto));
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
            for (const auto& node_proto : m_graph_proto->node())
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
            for (const auto& output : m_graph_proto->output())
            {
                results.emplace_back(get_ng_node_from_cache(output.name()));
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
            for (int i = 0; i < ng_node_vector.size(); ++i)
            {
                // Trailing optional outputs may not be specified in the ONNX model.
                // Other optional outputs should have name set to an empty string.
                if (i >= onnx_node.get_outputs_size())
                {
                    break;
                }

                ng_node_vector[i].get_node()->set_friendly_name(onnx_node.output(i));
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

        Subgraph::Subgraph(const ONNX_NAMESPACE::GraphProto& proto,
                           Model& model,
                           const Graph& parent_graph)
            : Graph(
                  proto,
                  model,
                  std::unique_ptr<SubgraphCache>(new SubgraphCache(parent_graph.get_graph_cache())))
        {
            std::vector<std::shared_ptr<ngraph::Node>> subgraph_root_nodes;
            const auto& outputs = as_result_vector(get_ng_outputs());
            for (auto& out : outputs)
            {
                subgraph_root_nodes.push_back(out);
            }
            const auto& params = get_ng_parameters();
            for (auto& param : params)
            {
                subgraph_root_nodes.push_back(param);
            }
            const auto subgraph_nodes = topological_sort(subgraph_root_nodes);

            const auto& parent_graph_parameters = parent_graph.get_ng_parameters();
            for (const auto& node : subgraph_nodes)
            {
                if (op::is_parameter(node))
                {
                    const auto sub_it = std::find(m_parameters.begin(), m_parameters.end(), node);
                    // not present as subgraph parameter
                    if (sub_it == m_parameters.end())
                    {
                        const auto parent_it = std::find(
                            parent_graph_parameters.begin(), parent_graph_parameters.end(), node);
                        if (parent_it != m_parameters.end())
                        {
                            m_parameters.push_back(*parent_it);
                        }
                    }
                }
            }
        }

    } // namespace onnx_import

} // namespace ngraph
