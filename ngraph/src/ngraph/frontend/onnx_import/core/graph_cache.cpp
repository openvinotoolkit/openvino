#include "graph_cache.hpp"
#include "except.hpp"
#include "provenance.hpp"
#include "tensor.hpp"
#include "utils/provenance_tag.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        GraphCache::GraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto)
        {
            // Process all initializers in the graph
            for (const auto& initializer_tensor : graph_proto.initializer())
            {
                if (initializer_tensor.has_name())
                {
                    Tensor tensor = Tensor{initializer_tensor};
                    m_initializers.emplace(initializer_tensor.name(), tensor);

                    // For each initializer create a Constant node and store it in cache
                    auto ng_constant = tensor.get_ng_constant();
                    add_provenance_tag_to_initializer(tensor, ng_constant);
                    m_graph_cache_map.emplace(initializer_tensor.name(), std::move(ng_constant));
                }
            }
        }

        const std::map<std::string, Tensor>& GraphCache::initializers() const
        {
            return m_initializers;
        }

        void GraphCache::add_provenance_tag_to_initializer(
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

        void GraphCache::add_node(const std::string& name, std::shared_ptr<ngraph::Node>&& node)
        {
            m_graph_cache_map[name] = std::move(node);
        }

        std::shared_ptr<ngraph::Node> GraphCache::get_node(const std::string& name) const
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

        SubgraphCache::SubgraphCache(const ONNX_NAMESPACE::GraphProto& graph_proto,
                                     const GraphCache* parent_graph_cache)
            : GraphCache(graph_proto)
            , m_parent_graph_cache{parent_graph_cache}
        {
            if (m_parent_graph_cache == nullptr)
            {
                throw ngraph_error("Parent graph cache is not initialized");
            }
        }

        std::shared_ptr<ngraph::Node> SubgraphCache::get_node(const std::string& name) const
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
