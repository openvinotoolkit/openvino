// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUBGRAPHS_HPP
#define SUBGRAPHS_HPP

#include <vector>
#include <unordered_set>
#include <utility>

#include <node.hpp>

#include <helpers/search.hpp>

#include <util/assert.hpp>
#include <util/algorithm.hpp>
#include <util/func_ref.hpp>

namespace ade
{
namespace details
{
template<typename Func>
inline void enumSubgraphsHelper(const NodeHandle& node, Func&& func)
{
    ASSERT(nullptr != node);
    for (auto edge: node->inEdges())
    {
        auto connected = edge->srcNode();
        if (func(connected, edge))
        {
            enumSubgraphsHelper(connected, std::forward<Func>(func));
        }
    }
    for (auto edge: node->outEdges())
    {
        auto connected = edge->dstNode();
        if (func(connected, edge))
        {
            enumSubgraphsHelper(connected, std::forward<Func>(func));
        }
    }
}
}

/// Scan graph starting from roots and split it to subgraphs, separated by
/// splitter.
/// Only nodes accessible from roots through non-splitters will be scaned
///
/// @param roots Scan roots
/// @param splitter Splitter functional object, take edge as parameter
/// must return true if two nodes should be splitted to different subgraphs
///
/// @return list of subgraphs
template<typename Nodes, typename Splitter>
inline std::vector<std::vector<NodeHandle>> splitSubgraphs(Nodes&& roots,
                                                           Splitter&& splitter)
{
    std::vector<std::vector<NodeHandle>> ret;
    if (!roots.empty())
    {
        std::unordered_set<NodeHandle, HandleHasher<Node>> markedNodes;
        for (auto&& node: roots)
        {
            if (util::contains(markedNodes, node))
            {
                continue;
            }
            markedNodes.insert(node);
            std::vector<NodeHandle> nodesToAdd;
            nodesToAdd.emplace_back(node);
            details::enumSubgraphsHelper(node,
                                         [&](const ade::NodeHandle& connected,
                                             const ade::EdgeHandle& edge)
            {
                if (util::contains(markedNodes, connected) || splitter(edge))
                {
                    return false;
                }
                markedNodes.insert(connected);
                nodesToAdd.emplace_back(connected);
                return true;
            });
            ret.emplace_back(std::move(nodesToAdd));
        }
    }
    return ret;
}

enum class SubgraphMergeDirection : int
{
    In,     /// Topologically previous node merged in current
    Out,    /// Topologically next node merged in current
};

namespace subgraphs
{
using NodesSet = std::unordered_set<NodeHandle, HandleHasher<Node>>;
}

/// Get node which we are merging to current node
///
/// @param edge - edge we are processing
/// @param dir - merge direction
///
/// @return node
inline NodeHandle getSrcMergeNode(const EdgeHandle& edge,
                               SubgraphMergeDirection dir)
{
    ASSERT(nullptr != edge);
    return (SubgraphMergeDirection::In == dir ?
                edge->dstNode() :
                edge->srcNode());
}

/// Get node to which we are merging
///
/// @param edge - edge we are processing
/// @param dir - merge direction
///
/// @return node
inline NodeHandle getDstMergeNode(const EdgeHandle& edge,
                               SubgraphMergeDirection dir)
{
    ASSERT(nullptr != edge);
    return (SubgraphMergeDirection::In == dir ?
                edge->srcNode() :
                edge->dstNode());
}

/// Assemble subgraph starting from specified root node.
/// Nodes are merged incrementally to root checking 2 conditions:
/// 1) Node can be merged to adjacent node
/// 2) Graph still topologically valid
///
/// @param root - root node from which we start subgraph assebly
/// @param mergeChecker - must return true if from provided in edge can be
/// merged
/// @param topoChecker - must return true if graph is still topologically valid,
/// takes 2 lists as parameters:
/// acceptedNodes - nodes already in graph.
/// rejectedNodes - nodes which will never be merged in subgraph
/// Any nodes not in these lists are not yet processed, they will get to one of
/// the list later
///
/// @return list of subgraph nodes, always contains root node
std::vector<NodeHandle> assembleSubgraph(
    const NodeHandle& root,
    util::func_ref<bool(const EdgeHandle&,
                        SubgraphMergeDirection)> mergeChecker,
    util::func_ref<bool(const subgraphs::NodesSet& acceptedNodes,
                        const subgraphs::NodesSet& rejectedNodes)> topoChecker);

/// Assemble multiple subgraphs starting from specified root nodes.
/// Subgraphs can intersect.
/// Nodes are merged incrementally to root checking 2 conditions:
/// 1) Node can be merged to adjacent node
/// 2) Graph still topologically valid
/// When we can't expand subgraph any further we will start another from next
/// unprocessed root node.
/// We will end when there are no uprocessed root nodes remaning.
///
/// @param roots - root node from which we can start subgraphs assebly
/// @param mergeChecker - must return true if from provided in edge can be
/// merged
/// @param topoChecker - must return true if graph is still topologically valid,
/// takes 2 lists as parameters:
/// acceptedNodes - nodes already in graph.
/// rejectedNodes - nodes which will never be merged in subgraph
/// Any nodes not in these lists are not yet processed, they will get to one of
/// the list later
///
/// @return list of subgraphs
template<typename NodesT, typename MChecker, typename TChecker>
inline std::vector<std::vector<NodeHandle>> selectSubgraphs(
        NodesT&& roots,
        MChecker&& mergeChecker,
        TChecker&& topoChecker)
{
    std::vector<std::vector<NodeHandle>> ret;
    subgraphs::NodesSet cannotBeRoots;
    for (auto&& root : roots)
    {
        if (!util::contains(cannotBeRoots, root))
        {
            auto nodes = assembleSubgraph(root,
                                          std::forward<MChecker>(mergeChecker),
                                          std::forward<TChecker>(topoChecker));
            if (!nodes.empty())
            {
                ret.emplace_back(std::move(nodes));
                for (auto&& node : ret.back())
                {
                    cannotBeRoots.insert(node);
                }
            }
        }
    }
    return ret;
}

struct SubgraphSelectResult final
{
    static constexpr const std::size_t NoSubgraph =
            static_cast<std::size_t>(-1);
    std::size_t index = NoSubgraph; /// Slected subgraph index or NoSubgraph
    bool continueSelect = false; /// Should we continue selection
};

/// Assemble multiple non-intersecting subgraphs starting from specified root
/// nodes.
/// Nodes are merged incrementally to root checking 2 conditions:
/// 1) Node can be merged to adjacent node
/// 2) Graph still topologically valid
/// When we can't expand subgraph any further we will start another from next
/// unprocessed root node.
/// On each iteration all available subgraphs are passed to selector functor,
/// selector must select one subgraph or abort selection.
/// We will end when there are no uprocessed root nodes remaning or when
/// selector return continueSelect == false.
///
/// @param roots - root node from which we can start subgraphs assebly
/// @param mergeChecker - must return true if from provided in edge can be
/// merged
/// @param topoChecker - must return true if graph is still topologically valid,
/// takes 2 lists as parameters:
/// acceptedNodes - nodes already in graph.
/// rejectedNodes - nodes which will never be merged in subgraph
/// Any nodes not in these lists are not yet processed, they will get to one of
/// the list later
/// @param selector - recieves list of available subgraphs,
/// must return SubgraphSelectResult
///
/// @return list of subgraphs
template<typename NodesT,
         typename MChecker,
         typename TChecker,
         typename Selector>
inline std::vector<std::vector<NodeHandle>> selectAllSubgraphs(
        NodesT&& roots,
        MChecker&& mergeChecker,
        TChecker&& topoChecker,
        Selector&& selector
        )
{
    std::vector<std::vector<NodeHandle>> ret;
    subgraphs::NodesSet availableNodes(std::begin(roots), std::end(roots));
    while(true)
    {
        auto subgraphs = selectSubgraphs(
                             availableNodes, [&](
                             const EdgeHandle& edge,
                             SubgraphMergeDirection dir)
        {
            auto dstNode = getDstMergeNode(edge, dir);
            if (!util::contains(availableNodes, dstNode))
            {
                return false;
            }
            return mergeChecker(edge, dir);
        },
        std::forward<TChecker>(topoChecker));
        if (subgraphs.empty())
        {
            break;
        }
        const SubgraphSelectResult res = selector(subgraphs);
        if (SubgraphSelectResult::NoSubgraph == res.index)
        {
            break;
        }
        ASSERT(res.index < subgraphs.size());
        ret.emplace_back(std::move(subgraphs[res.index]));
        if (res.continueSelect)
        {
            for (auto&& node : ret.back())
            {
                ASSERT(nullptr != node);
                availableNodes.erase(node);
            }
        }
        else
        {
            break;
        }
    }
    return ret;
}

/// Enumerate all paths throuh output edges from src node to dst node
///
/// @param src - Source node
/// @param dst - Destination node
/// @param visitor - visitor receives paths, should return true to abort
/// iteration
void findPaths(const NodeHandle& src, const NodeHandle& dst,
               util::func_ref<bool(const std::vector<NodeHandle>&)> visitor);

/// Enumerate all paths throuh output edges from node1 node to node2 node and
/// from node2 to node1
///
/// @param node1 - Source or destination node
/// @param node2 - Source or destination node
/// @param visitor - visitor receives paths, should return true to abort
/// iteration
template<typename Visitor>
inline void findBiPaths(const NodeHandle& node1, const NodeHandle& node2,
                        Visitor&& visitor)
{
    bool found = false;
    (void)found; // Silence klocwork warning
    findPaths(node1, node2, [&](const std::vector<NodeHandle>& path)
    {
        found = visitor(path);
        return found;
    });
    if (!found)
    {
        findPaths(node2, node1, std::forward<Visitor>(visitor));
    }
}

/// This class checks subgraphs for self references
struct SubgraphSelfReferenceChecker final
{
    /// Check if there is self reference in subgraph, when some nodes if some
    /// nodes in subgraph are reachaeble by other nodes in subgraph through
    /// rejected nodes.
    /// Assuming nodes are never removed from rejected list.
    /// Accepts two node lists, first for nodes in subgraph,
    /// second for nodes not in subgraphs, nodes not included in lists
    /// are ignored.
    ///
    /// @param acceptedNodes - subgraph to check
    /// @param rejectedNodes - nodes not in subgraphs
    ///
    /// @return true if there is a self reference in subgraph
    bool operator()(const subgraphs::NodesSet& acceptedNodes,
                    const subgraphs::NodesSet& rejectedNodes);

    /// Resets internal cache
    /// You must call this function if you are going to remove nodes from
    /// rejected list.
    void reset();

    /// Contructor
    ///
    /// @param nodes - all nodes in graph
    template<typename Nodes>
    SubgraphSelfReferenceChecker(Nodes&& nodes)
    {
        transitiveClosure(std::forward<Nodes>(nodes),
                          [this](const NodeHandle& src,
                                 const NodeHandle& dst)
        {
            m_connections[src].insert(dst);
        });
    }

private:
    using Connections =
    std::unordered_map<ade::NodeHandle,
                       std::unordered_set<ade::NodeHandle, HandleHasher<Node>>,
                       HandleHasher<Node>>;

    Connections m_connections;

    struct Hasher final
    {
        std::size_t operator()(
                const std::pair<ade::NodeHandle,ade::NodeHandle>& value) const;
    };
    using Cache = std::unordered_map<std::pair<ade::NodeHandle,ade::NodeHandle>,
                                     subgraphs::NodesSet,
                                     Hasher>;
    Cache m_cache;

    void updateCache(const std::pair<ade::NodeHandle,ade::NodeHandle>& key);
};

}

#endif // SUBGRAPHS_HPP
