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

#pragma once

#include <onnx/onnx_pb.h>
#include <string>
#include <vector>

#include "default_opset.hpp"
#include "graph_cache.hpp"
#include "model.hpp"
#include "ngraph/op/parameter.hpp"
#include "operator_set.hpp"
#include "value_info.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Graph
        {
        public:
            Graph(const ONNX_NAMESPACE::GraphProto& proto, Model& model);
            const std::vector<Node>& get_nodes() const { return m_nodes; }
            const std::vector<ValueInfo>& get_inputs() const { return m_inputs; }
            const std::vector<ValueInfo>& get_outputs() const { return m_outputs; }
            NodeVector get_ng_outputs() const;
            const ParameterVector& get_ng_parameters() const { return m_parameters; }
            bool is_node_in_cache(const std::string& name) const;
            std::shared_ptr<ngraph::Node> get_ng_node_from_cache(const std::string& name) const
            {
                return m_cache.get_node(name);
            }
            const std::string& get_name() const { return m_graph_proto->name(); }
            NodeVector make_ng_nodes(const Node& onnx_node) const;
            const GraphCache& get_graph_cache() const;

        protected:
            Graph(const ONNX_NAMESPACE::GraphProto& proto, Model& model, GraphCache&& cache);

            void set_friendly_names(const Node& onnx_node, const NodeVector& ng_node_vector) const;

            void add_provenance_tag_to_initializer(
                const Tensor& initializer, std::shared_ptr<default_opset::Constant> node) const;

            void add_provenance_tag_to_input(const ValueInfo& input,
                                             std::shared_ptr<ngraph::Node> node) const;

            void add_provenance_tags(const Node& onnx_node, const NodeVector& ng_node_vector) const;

        private:
            const ONNX_NAMESPACE::GraphProto* m_graph_proto;
            GraphCache&& m_cache;
            std::vector<Node> m_nodes;
            std::vector<ValueInfo> m_inputs;
            std::vector<ValueInfo> m_outputs;
            ParameterVector m_parameters;
            std::map<std::string, Tensor> m_initializers;
            Model* m_model;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Graph& graph)
        {
            return (outs << "<Graph: " << graph.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
