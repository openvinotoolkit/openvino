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

#include <memory>
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>

#include "ngraph/op/parameter.hpp"
#include "onnx_import/core/graph_cache.hpp"
#include "onnx_import/core/model.hpp"
#include "onnx_import/core/operator_set.hpp"
#include "onnx_import/core/value_info.hpp"
#include "onnx_import/default_opset.hpp"

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
            OutputVector get_ng_outputs() const;
            const ParameterVector& get_ng_parameters() const { return m_parameters; }
            bool is_node_in_cache(const std::string& name) const;
            Output<ngraph::Node> get_ng_node_from_cache(const std::string& name) const;
            const std::string& get_name() const { return m_graph_proto->name(); }
            OutputVector make_ng_nodes(const Node& onnx_node) const;
            const GraphCache& get_graph_cache() const;
            const OpsetImports& get_opset_imports() const;

        protected:
            Graph(const ONNX_NAMESPACE::GraphProto& proto,
                  Model& model,
                  std::unique_ptr<GraphCache>&& cache);

            void set_friendly_names(const Node& onnx_node,
                                    const OutputVector& ng_node_vector) const;

            void add_provenance_tag_to_initializer(
                const Tensor& initializer, std::shared_ptr<default_opset::Constant> node) const;

            void add_provenance_tag_to_input(const ValueInfo& input,
                                             std::shared_ptr<ngraph::Node> node) const;

            void add_provenance_tags(const Node& onnx_node,
                                     const OutputVector& ng_node_vector) const;

        protected:
            ParameterVector m_parameters;
            std::unique_ptr<GraphCache> m_cache;

        private:
            const ONNX_NAMESPACE::GraphProto* m_graph_proto;
            std::vector<Node> m_nodes;
            std::vector<ValueInfo> m_inputs;
            std::vector<ValueInfo> m_outputs;
            Model* m_model;
        };

        /// \brief      Representation of ONNX subgraph. It is used for example by ONNX Loop op.
        ///             It has access for initializers both from subgraph and from parent graph
        ///             cache.
        class Subgraph : public Graph
        {
        public:
            /// \brief      Subgraph a GraphCache class object.
            ///
            /// \param[in]  proto          The ONNX protobuf graph representation.
            /// \param[in]  model          The ONNX model object.
            /// \param[in]  parent_graph   The reference to the parent graph.
            Subgraph(const ONNX_NAMESPACE::GraphProto& proto,
                     Model& model,
                     const Graph& parent_graph);

            /// \brief      Return outputs which are on the edge the subgraph and the parent graph.
            /// \return     Vector of edge nodes from parent scope.
            const std::vector<Output<ngraph::Node>> get_outputs_from_parent() const;

        private:
            std::vector<Output<ngraph::Node>> m_outputs_from_parent;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Graph& graph)
        {
            return (outs << "<Graph: " << graph.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
