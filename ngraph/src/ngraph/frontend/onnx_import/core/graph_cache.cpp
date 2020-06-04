#include "graph_cache.hpp"
#include "except.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        // TODO DOC
        GraphCache::GraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto)
        {
            // Process all initializers in the graph
            for (const auto& initializer_tensor : graph_proto.initializer())
            {
                if (initializer_tensor.has_name())
                {
                    Tensor tensor = Tensor{initializer_tensor};
                    // For each initializer, create a Constant node and store in cache
                    auto ng_constant = tensor.get_ng_constant();
                    m_graph_cache_map.emplace(initializer_tensor.name(), std::move(ng_constant));
                }
            }
        }

        void GraphCache::set_node(const std::string& name, std::shared_ptr<ngraph::Node>&& node)
        {
            // TODO Add exception throwing
            m_graph_cache_map[name] = std::move(node);
        }

        std::shared_ptr<ngraph::Node> GraphCache::get_node(const std::string& name) const
        {
            try
            {
                return m_graph_cache_map.at(name);
            }
            catch(const std::out_of_range)
            {
                throw ngraph_error(name + " node not found in graph cache");
            }
        }

        bool GraphCache::contains(const std::string& node_name) const
        {
            return (m_graph_cache_map.count(node_name) > 0);
        }



        SubgraphCache::SubgraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto, const GraphCache& parent_graph_cache)
            : GraphCache(graph_proto)
            , m_parent_graph_cache{parent_graph_cache}
        {
        }

        // TODO Not override
        void SubgraphCache::set_node(const std::string& name, std::shared_ptr<ngraph::Node>&& node)
        {
            // present in subgraph
            if(GraphCache::contains(name))
            {
                // TODO use [] ?
                m_graph_cache_map[name] = std::move(node);
            }
            /*else // defined in parent graph scope
            {
               m_parent_graph_cache.set_node(name, std::move(node));
            }*/
        }

        std::shared_ptr<ngraph::Node> SubgraphCache::get_node(const std::string& name) const
        {
            // present in subgraph
            if(GraphCache::contains(name))
            {
                // TODO use [] ?
                return GraphCache::get_node(name);
            }
            else // defined in parent graph scope
            {
               return m_parent_graph_cache.get_node(name);
            }
        }

        bool SubgraphCache::contains(const std::string& node_name) const
        {
            return GraphCache::contains(node_name) || m_parent_graph_cache.contains(node_name);
        }

    } // namespace onnx_import
} // namespace ngraph
