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

#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <string>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief      GraphCache stores and provides access to ONNX graph initializers.
        class GraphCache
        {
        public:
            /// \brief      Constructs a GraphCache class object.
            ///
            /// \param[in]  graph_proto       ONNX protobuf graph representation.
            GraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto);

            /// \brief      Add node to the cache or override the existing one.
            ///
            /// \note       GraphCahce takes ownership of the node.
            ///
            /// \param[in]  name       The name of node added to the cache.
            /// \param[in]  node       The node added to the cache.
            void add_node(const std::string& name, std::shared_ptr<ngraph::Node>&& node);

            /// \brief      Get the node from the cache
            ///
            /// \note       If the node is not found the ngraph_error exception is thrown.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     The node named `name`.
            virtual std::shared_ptr<ngraph::Node> get_node(const std::string& name) const;

            /// \brief      Return true if the node named `name` exist in the cache.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     true if the node named `name` exist in the cache, false otherwise.
            virtual bool contains(const std::string& name) const;

            /// \brief      Return the map of graph initializers.
            ///
            /// \return     The map of graph initializers from the graph_proto.
            const std::map<std::string, Tensor>& initializers() const;

        private:
            /// \brief      Add provenance tag to initializer which helps during debuging.
            ///
            /// \param[in]  initializer   The tensor initializer used to create provenance tag.
            /// \param[in]  node          The node to which provenance tag is added.
            void add_provenance_tag_to_initializer(
                const Tensor& initializer, std::shared_ptr<default_opset::Constant> node) const;

            std::map<std::string, std::shared_ptr<ngraph::Node>> m_graph_cache_map;
            std::map<std::string, Tensor> m_initializers;
        };

        class SubgraphCache : public GraphCache
        {
        public:
            /// \brief      Constructs a SubgraphCache class object.
            ///
            /// \param[in]  graph_proto          ONNX protobuf graph representation.
            /// \param[in]  parent_graph_cache   The shared pointer to the parent graph.
            SubgraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto,
                          const std::shared_ptr<GraphCache> parent_graph_cache);

            /// \brief      Get the node from the cache (subgraph or parent graph)
            ///
            /// \note       If the node is not found the ngraph_error exception is thrown.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     The node named `name` from subgraph (as first) or from parent graph.
            std::shared_ptr<ngraph::Node> get_node(const std::string& name) const override;

            /// \brief      Return true if the node named `name` exist in the cache.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     true if the node named `name` exist in the cache
            ///             (subgraph or parent graph), false otherwise.
            bool contains(const std::string& name) const override;

        private:
            const std::shared_ptr<GraphCache> m_parent_graph_cache;
        };

    } // namespace onnx_import
} // namespace ngraph
