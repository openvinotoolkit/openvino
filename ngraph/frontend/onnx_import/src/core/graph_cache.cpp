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

#include <ngraph/except.hpp>
#include <onnx_import/core/graph_cache.hpp>

namespace ngraph
{
    namespace onnx_import
    {
        void GraphCache::emplace_node(const std::string& name, Output<ngraph::Node>&& node)
        {
            m_graph_cache_map[name] = std::move(node);
        }

        Output<ngraph::Node> GraphCache::get_node(const std::string& name) const
        {
            try
            {
                return m_graph_cache_map.at(name);
            }
            catch (const std::out_of_range&)
            {
                throw ngraph_error(name + " node not found in graph cache");
            }
        }

        bool GraphCache::contains(const std::string& name) const
        {
            return (m_graph_cache_map.count(name) > 0);
        }

        SubgraphCache::SubgraphCache(const GraphCache& parent_graph_cache)
            : m_parent_graph_cache{&parent_graph_cache}
        {
            if (m_parent_graph_cache == nullptr)
            {
                throw ngraph_error("Parent graph cache is not initialized");
            }
        }

        Output<ngraph::Node> SubgraphCache::get_node(const std::string& name) const
        {
            // present in subgraph scope
            if (GraphCache::contains(name))
            {
                return GraphCache::get_node(name);
            }
            else // present in parent graph scope
            {
                return m_parent_graph_cache->get_node(name);
            }
        }

        bool SubgraphCache::contains(const std::string& name) const
        {
            // the node is in subgraph or in parent graph scope
            return GraphCache::contains(name) || m_parent_graph_cache->contains(name);
        }

    } // namespace onnx_import
} // namespace ngraph
