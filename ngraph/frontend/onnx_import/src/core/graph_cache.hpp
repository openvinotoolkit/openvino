// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /// \brief  Enum which determines scope (visibility) of nodes in GraphCache.
        enum class NodeScope
        {
            // in parent graph scope
            ParentGraph = 1,
            // in subgraph scope
            SubGraph,
            // not available at all
            Lack
        };

        /// \brief      GraphCache stores and provides access to ONNX graph initializers.
        class GraphCache
        {
        public:
            /// \brief      Add node to the cache or override the existing one.
            ///
            /// \note       GraphCache takes ownership of the node.
            ///
            /// \param[in]  name       The name of node added to the cache.
            /// \param[in]  node       The node added to the cache.
            void emplace_node(const std::string& name, Output<ngraph::Node>&& node);

            /// \brief      Remove node from the cache
            ///
            /// \param[in]  name       The name of node to be removed
            void remove_node(const std::string& name);

            /// \brief      Get the node from the cache
            ///
            /// \note       If the node is not found the ngraph_error exception is thrown.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     The node named `name`.
            virtual Output<ngraph::Node> get_node(const std::string& name) const;

            /// \brief      Return true if the node named `name` exist in the cache.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     true if the node named `name` exist in the cache, false otherwise.
            virtual bool contains(const std::string& name) const;

            /// \brief      Return NodeScope enum which determines scope of the node.
            /// \note       If the method is called on GraphCache the ParentGraph enum
            ///             value is retunred always.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     SubGraph if node belongs to SubgraphCache, ParentGraph if
            ///             is avalible in parent_graph_cache, otherwise Lack
            virtual NodeScope node_scope(const std::string& name) const;

            virtual ~GraphCache() = default;

        private:
            std::map<std::string, Output<ngraph::Node>> m_graph_cache_map;
        };

        class SubgraphCache : public GraphCache
        {
        public:
            /// \brief      Constructs a SubgraphCache class object.
            ///
            /// \param[in]  parent_graph_cache   The reference to the parent graph.
            SubgraphCache(const GraphCache& parent_graph_cache);

            /// \brief      Get the node from the cache (subgraph or parent graph)
            ///
            /// \note       If the node is not found the ngraph_error exception is thrown.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     The node named `name` from subgraph (as present) or from parent graph.
            Output<ngraph::Node> get_node(const std::string& name) const override;

            /// \brief      Return true if the node named `name` exist in the cache.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     true if the node named `name` exist in the cache
            ///             (subgraph or parent graph), false otherwise.
            bool contains(const std::string& name) const override;

            /// \brief      Return NodeScope enum which determines scope of the node.
            ///
            /// \param[in]  name       The name of the node.
            ///
            /// \return     SubGraph if the node belongs to SubgraphCache, ParentGraph if
            ///             is avalible in parent_graph_cache, otherwise Lack
            NodeScope node_scope(const std::string& name) const override;

        private:
            const GraphCache* m_parent_graph_cache;
        };

    } // namespace onnx_import
} // namespace ngraph
