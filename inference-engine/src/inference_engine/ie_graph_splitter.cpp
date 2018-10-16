// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_graph_splitter.hpp"

#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <string>

#include <ade_util.hpp>

#include <typed_graph.hpp>
#include <helpers/subgraphs.hpp>

#include <util/filter_range.hpp>
#include <util/iota_range.hpp>

namespace InferenceEngine {

namespace {
class ISplitChecker {
public:
    struct GraphSelectionResult final {
        static const constexpr std::size_t NoGraph
            = static_cast<std::size_t>(-1);

        std::size_t selectedGraph = NoGraph;
        bool continueSelect = false;
    };

    virtual ~ISplitChecker() = default;
    virtual GraphSelectionResult selectSubgraph(
            const std::vector<LayersSet>& subgraphs) = 0;
};

class DefaultSplitChecker : public ISplitChecker {
public:
    // ISplitChecker interface
    GraphSelectionResult selectSubgraph(const std::vector<LayersSet>& subgraphs) override;
};
}  // namespace

std::vector<LayersSet> splitGraph(ICNNNetwork& network,
        const std::vector<std::string>& plugins) {
    assert(!plugins.empty());
    ade::Graph gr;
    ade::TypedGraph<CNNLayerMetadata> tgr(gr);

    std::vector<LayersSet> tempSubgraphs;
    LayersSet tempSet1;
    LayersSet tempSet2;

    translateNetworkToAde(gr, network);
    std::size_t currentChecker = 0;

    DefaultSplitChecker checker;

    auto getChecker = [&]() {
        assert(currentChecker < plugins.size());
        return &checker;
    };

    auto getAffinity = [&]()->const std::string& {
        assert(currentChecker < plugins.size());
        return plugins[currentChecker];
    };

    auto nodes = gr.nodes();
    ade::subgraphs::NodesSet availableNodes(nodes.begin(), nodes.end());
    std::vector<LayersSet> finalSubgraphs;
    ade::SubgraphSelfReferenceChecker cycleChecker(nodes);
    while (!availableNodes.empty()) {
        auto subgraphs = ade::selectSubgraphs(
                             util::filter(util::toRange(availableNodes),
                                          [&](const ade::NodeHandle& node) {
            assert(nullptr != node);
            auto layer = tgr.metadata(node).get<CNNLayerMetadata>().layer;
            assert(nullptr != layer);
            return layer->affinity == getAffinity();
        }),
                             [&](
                             const ade::EdgeHandle& edge,
                             ade::SubgraphMergeDirection dir) {
            assert(nullptr != edge);
            auto dstNode = ade::getDstMergeNode(edge, dir);
            assert(nullptr != dstNode);
            if (!util::contains(availableNodes, dstNode)) {
                return false;
            }
            auto srcNode = ade::getSrcMergeNode(edge, dir);
            assert(nullptr != srcNode);
            auto srcLayer = tgr.metadata(srcNode).get<CNNLayerMetadata>().layer;
            auto dstLayer = tgr.metadata(dstNode).get<CNNLayerMetadata>().layer;
            assert(nullptr != srcLayer);
            assert(nullptr != dstLayer);
            return srcLayer->affinity == dstLayer->affinity;
        },
                             [&](
                             const ade::subgraphs::NodesSet& acceptedNodes,
                             const ade::subgraphs::NodesSet& rejectedNodes) {
            if (cycleChecker(acceptedNodes, rejectedNodes)) {
                return false;
            }
            return true;
        });

        if (!subgraphs.empty()) {
            if (plugins.size() == currentChecker) {
                THROW_IE_EXCEPTION << "Some nodes weren't assigned to plugin";
            }

            tempSubgraphs.clear();
            for (auto&& subgraph : subgraphs) {
                assert(!subgraph.empty());
                tempSet1.clear();
                for (auto&& node : subgraph) {
                    assert(nullptr != node);
                    auto layer = tgr.metadata(node).get<CNNLayerMetadata>().layer;
                    assert(nullptr != layer);
                    tempSet1.insert(layer);
                }
                tempSubgraphs.emplace_back(std::move(tempSet1));
            }
            auto result = getChecker()->selectSubgraph(tempSubgraphs);
            const auto selected = result.selectedGraph;
            if (ISplitChecker::GraphSelectionResult::NoGraph !=
                    selected) {
                assert(selected < subgraphs.size());
                finalSubgraphs.emplace_back(std::move(tempSubgraphs[selected]));

                for (auto&& node : subgraphs[selected]) {
                    availableNodes.erase(node);
                }

                if (result.continueSelect) {
                    continue;
                }
            }
        }
        ++currentChecker;
    }

    return finalSubgraphs;
}

ISplitChecker::GraphSelectionResult DefaultSplitChecker::selectSubgraph(
        const std::vector<LayersSet>& subgraphs) {
    assert(!subgraphs.empty());
    std::size_t index = 0;
    auto maxSize = subgraphs[0].size();
    for (auto i : util::iota(std::size_t(1), subgraphs.size())) {
        auto size = subgraphs[i].size();
        if (size > maxSize) {
            index = 1;
            maxSize = size;
        }
    }
    GraphSelectionResult ret;
    ret.selectedGraph = index;
    ret.continueSelect = true;
    return ret;
}

namespace {
struct SubgraphDesc {
    std::size_t topoIndex = static_cast<std::size_t>(-1);
    std::unordered_set<std::size_t> dependsOn;
};

void topoVisitSubgraph(std::vector<SubgraphDesc>& subgraphs,
                       SubgraphDesc& subgraph,
                       std::size_t& topoIndex) {
    if (subgraph.topoIndex != static_cast<std::size_t>(-1)) {
        assert(subgraph.topoIndex < topoIndex);
        return;
    }

    for (auto&& dep : subgraph.dependsOn) {
        topoVisitSubgraph(subgraphs, subgraphs[dep], topoIndex);
    }
    subgraph.topoIndex = topoIndex;
    ++topoIndex;
}
}  // namespace

void sortSubgraphs(std::vector<LayersSet>& subgraphs) {
    std::vector<SubgraphDesc> descs(subgraphs.size());

    for (auto i : util::iota(subgraphs.size())) {
        auto& subgraph = subgraphs[i];
        assert(!subgraph.empty());
        for (auto&& layer : subgraph) {
            assert(nullptr != layer);
            for (auto&& dataIt : layer->insData) {
                auto data = dataIt.lock();
                assert(nullptr != data);
                auto prevLayer = data->creatorLayer.lock();
                if (nullptr != prevLayer) {
                    for (auto j : util::iota(subgraphs.size())) {
                        if (i != j) {
                            if (util::contains(subgraphs[j], prevLayer)) {
                                descs[i].dependsOn.insert(j);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    {
        std::size_t topoIndex = 0;
        for (auto&& desc : descs) {
            topoVisitSubgraph(descs, desc, topoIndex);
        }
        assert(subgraphs.size() == topoIndex);
    }

    std::vector<LayersSet> ret(subgraphs.size());
    for (auto i : util::iota(subgraphs.size())) {
        assert(i < descs.size());
        auto& desc = descs[i];
        auto topoIndex = desc.topoIndex;
        assert(topoIndex != static_cast<std::size_t>(-1));
        assert(topoIndex < ret.size());
        assert(!subgraphs[i].empty());
        ret[topoIndex] = std::move(subgraphs[i]);
    }
    subgraphs = std::move(ret);
}

}  // namespace InferenceEngine
