// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "matchers/subgraph/subgraph.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class FusedNamesExtractor final : public SubgraphExtractor {
public:
    FusedNamesExtractor(const std::string& device = "");

    std::vector<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &modele) override;

protected:
    std::unordered_set<std::string> extract_not_trasformed_node_names(const std::shared_ptr<ov::Model>& model);
    void set_target_device(const std::string& _device);

    std::string device;
    
    // possible solution to imrove pattern extraction
    // struct NodeDescriptor {
    //     std::shared_ptr<ov::Node> node;
    //     std::unordered_set<size_t> input_idx, output_idx;
    //     size_t subgraph_id = std::numeric_limits<size_t>::max();

    //     NodeDescriptor(const std::shared_ptr<ov::Node>& in_node) : node(in_node) {}

    //     bool is_defined() {
    //         return subgraph_id != std::numeric_limits<size_t>::max();
    //     }
    // };
    
    // std::vector<NodeDescriptor> extract_transformed_nodes(const std::shared_ptr<ov::Model>& model);
    // labeled subgraphs: {subgraph_id, NodeVector}
    // std::unordered_map<size_t, ov::NodeVector> label_subgrapohs(std::vector<NodeDescriptor>& transformed_ops);
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
