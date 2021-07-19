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

            virtual ~GraphCache() = default;

        private:
            std::map<std::string, Output<ngraph::Node>> m_graph_cache_map;
        };
    } // namespace onnx_import
} // namespace ngraph
