// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matchers/subgraph/subgraph.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class RepeatPatternExtractor final : public SubgraphExtractor {
private:
    using InputVector = std::vector<ov::Input<ov::Node>>;
    using OutputVector = std::vector<ov::Output<ov::Node>>;

public:
    using PatternBorders = std::pair<InputVector, OutputVector>;
    ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();

    std::vector<std::vector<PatternBorders>>
    get_repeat_pattern_borders(const std::shared_ptr<ov::Model> &model);
    std::vector<std::vector<ov::NodeVector>>
    get_repeat_node_vectors(const std::shared_ptr<ov::Model> &model);

    void set_recursive_extraction(bool _is_recursive_extraction);
    std::vector<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) override;

    std::vector<ov::NodeVector>
    get_node_vector(const ov::NodeVector& start_node_vec);

protected:
    // {subgraph, node_vector, input_info}
    using ExtractedRepeatPattern = std::tuple<std::shared_ptr<ov::Model>, ov::NodeVector, std::map<std::string, ov::conformance::InputInfo>>;
    bool is_recursive_extraction = true;

    std::list<std::vector<ExtractedRepeatPattern>>
    find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                         bool is_save_borders_only = false);
    void update_extractor_cache(std::list<std::vector<ExtractedRepeatPattern>>& extracted_patterns,
                                std::list<std::vector<ExtractedRepeatPattern>>& secondary_extracted_patterns);
    void update_extractor_cache(std::list<std::vector<ExtractedRepeatPattern>>& extracted_patterns,
                                const std::shared_ptr<ov::Model>& pattern,
                                const ov::NodeVector& pattern_node_vector,
                                const std::map<std::string, ov::conformance::InputInfo>& in_info);

};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
