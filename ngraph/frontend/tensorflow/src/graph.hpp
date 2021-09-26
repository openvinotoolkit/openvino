// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tensorflow_frontend/model.hpp>
#include <tensorflow_frontend/place.hpp>
#include "decoder_new.hpp"
#include "graph.pb.h"

namespace tensorflow {
class GraphDef;
class NodeDef;
}  // namespace tensorflow

namespace tensorflow {
namespace ngraph_bridge {

class GraphIteratorProto : public ::ngraph::frontend::GraphIterator {
    std::vector<const ::tensorflow::NodeDef*> nodes;
    size_t node_index = 0;

public:
    GraphIteratorProto(const ::tensorflow::GraphDef* _graph) {
        // TODO: Sort topologicaly nodes from the graph
        nodes.resize(_graph->node_size());
        for (size_t i = 0; i < nodes.size(); ++i)
            nodes[i] = &_graph->node(i);
    }

    GraphIteratorProto(const std::vector<std::shared_ptr<::tensorflow::NodeDef>>& _sorted_nodes) {
        nodes.resize(_sorted_nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
            nodes[i] = _sorted_nodes[i].get();
    }

    /// Set iterator to the start position
    virtual void reset() override {
        node_index = 0;
    }

    virtual size_t size() const override {
        return nodes.size();
    }

    /// Moves to the next node in the graph
    virtual void next() override {
        node_index++;
    }

    virtual bool is_end() const override {
        return node_index >= nodes.size();
    }

    /// Return NodeContext for the current node that iterator points to
    virtual std::shared_ptr<ngraph::frontend::DecoderBase> get_new() const override {
        return std::make_shared<::ngraph::frontend::DecoderTFProto>(nodes[node_index]);
    }
};

}  // namespace ngraph_bridge
}  // namespace tensorflow
