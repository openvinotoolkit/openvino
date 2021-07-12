// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <core/graph_cache.hpp>
#include <ngraph/except.hpp>

namespace ngraph
{
    namespace onnx_import
    {
        void GraphCache::emplace_node(const std::string& name, Output<ngraph::Node>&& node)
        {
            m_graph_cache_map[name] = std::move(node);
        }

        void GraphCache::remove_node(const std::string& name)
        {
            auto it = m_graph_cache_map.find(name);
            if (it != m_graph_cache_map.end())
            {
                m_graph_cache_map.erase(it);
            }
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
    } // namespace onnx_import
} // namespace ngraph
