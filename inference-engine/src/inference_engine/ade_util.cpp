// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ade_util.hpp"

#include <unordered_map>
#include <utility>

#include <ie_icnn_network.hpp>
#include <ie_util_internal.hpp>
#include <ie_layers.h>

#include <ade/util/algorithm.hpp>
#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>

namespace InferenceEngine {
namespace {
using VisitedLayersMap = std::unordered_map<CNNLayer::Ptr, ade::NodeHandle>;
using TGraph = ade::TypedGraph<CNNLayerMetadata>;

void translateVisitLayer(VisitedLayersMap& visited,
                TGraph& gr,
                const ade::NodeHandle& prevNode,
                const CNNLayer::Ptr& layer) {
    assert(nullptr != layer);;
    assert(!ade::util::contains(visited, layer));
    auto node = gr.createNode();
    gr.metadata(node).set(CNNLayerMetadata{layer});
    if (nullptr != prevNode) {
        gr.link(prevNode, node);
    }
    visited.insert({layer, node});
    for (auto&& data : layer->outData) {
        for (auto&& layerIt : data->getInputTo()) {
            auto nextLayer = layerIt.second;
            auto it = visited.find(nextLayer);
            if (visited.end() == it) {
                translateVisitLayer(visited, gr, node, nextLayer);
            } else {
                gr.link(node, it->second);
            }
        }
    }
}
}  // namespace

void translateNetworkToAde(ade::Graph& gr, ICNNNetwork& network) {
    TGraph tgr(gr);
    VisitedLayersMap visited;
    for (auto& data : getRootDataObjects(network)) {
        assert(nullptr != data);
        for (auto& layerIt : data->getInputTo()) {
            auto layer = layerIt.second;
            assert(nullptr != layer);
            if (!ade::util::contains(visited, layer)) {
                translateVisitLayer(visited, tgr, nullptr, layer);
            }
        }
    }
}

const char* CNNLayerMetadata::name() {
    return "CNNLayerMetadata";
}

}  // namespace InferenceEngine
