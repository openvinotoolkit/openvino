// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/subgraph/subgraph.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

// note: please set any specific parameters related to graph comparation by `ModelComparetor::get()`
// for example node attributes match or shape strict comparation
class RepeatPatternExtractor final : public SubgraphExtractor {
private:
    using InputVector = std::vector<ov::Input<ov::Node>>;
    using OutputVector = std::vector<ov::Output<ov::Node>>;
    using NodePair = std::pair<std::shared_ptr<ov::Node>, std::vector<size_t>>;

public:
    using PatternBorders = std::pair<InputVector, OutputVector>;
    ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();

    std::vector<std::vector<PatternBorders>>
    get_repeat_pattern_borders(const std::shared_ptr<ov::Model> &model);
    std::vector<std::vector<ov::NodeVector>>
    get_repeat_node_vectors(const std::shared_ptr<ov::Model> &model);

    std::vector<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) override;

    // minimal size of extracted subgraph
    void set_min_graph_size(size_t _min_graph_size) {
        min_graph_size = _min_graph_size;
    }
    // cut graphs by matched nodes
    void set_split_by_matched_nodes(bool _is_split_by_matched_nodes) {
        is_split_by_matched_nodes = _is_split_by_matched_nodes;
    }
    // recursive extraction from extracted subgraphs
    void set_recursive_extraction(bool _is_recursive_extraction) {
        is_recursive_extraction = _is_recursive_extraction;
    }

protected:
    // {subgraph, node_vector, input_info}
    using ExtractedRepeatPattern = std::tuple<std::shared_ptr<ov::Model>, ov::NodeVector, std::map<std::string, ov::conformance::InputInfo>>;
    size_t min_graph_size = 2;
    bool is_split_by_matched_nodes = false, is_recursive_extraction = false;

    // find repeat patterns in model
    std::list<std::vector<ExtractedRepeatPattern>>
    find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                         bool is_save_borders_only = false);
    void update_extractor_cache(std::list<std::vector<ExtractedRepeatPattern>>& extracted_patterns,
                                std::list<std::vector<ExtractedRepeatPattern>>& secondary_extracted_patterns);
    void update_extractor_cache(std::list<std::vector<ExtractedRepeatPattern>>& extracted_patterns,
                                const std::shared_ptr<ov::Model>& pattern,
                                const std::vector<ov::NodeVector>& pattern_node_vector,
                                const std::map<std::string, ov::conformance::InputInfo>& in_info);
    // extract repeated patterns by start_node
    std::vector<std::vector<ov::NodeVector>>
    get_patterns_by_nodes(const std::vector<size_t>& start_op_vec,
                          const ov::NodeVector& ordered_ops);

};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
