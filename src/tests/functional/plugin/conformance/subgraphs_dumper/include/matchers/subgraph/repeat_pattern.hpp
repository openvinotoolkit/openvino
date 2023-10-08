// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "matchers/subgraph/subgraph.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class RepeatPatternExtractor final : public SubgraphExtractor {
private:
    using InputVector = std::vector<ov::Input<ov::Node>>;
    using OutputVector = std::vector<ov::Output<ov::Node>>;

public:
    using PatternBorders = std::pair<InputVector, OutputVector>;

    std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model) override;

    std::vector<std::vector<PatternBorders>>
    get_repeat_pattern_borders(const std::shared_ptr<ov::Model> &model);

    std::vector<std::vector<ov::NodeVector>>
    get_repeat_node_vectors(const std::shared_ptr<ov::Model> &model);

    void set_recursive_extraction(bool _is_recursive_extraction) { is_recursive_extraction = _is_recursive_extraction; }
    void set_model_match_coefficient(float _match_coefficient) { model_comparator->set_match_coefficient(_match_coefficient); }

private:
    using ExtractedRepeatPattern = std::pair<std::map<std::shared_ptr<ov::Model>, ov::NodeVector>, std::map<std::string, InputInfo>>;

    ModelComparator::Ptr model_comparator = ModelComparator::get();
    bool is_recursive_extraction = true;

    std::vector<ExtractedRepeatPattern>
    find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                         bool is_save_borders_only = false);
    void update_extractor_cache(std::vector<ExtractedRepeatPattern>& extracted_patterns,
                                std::vector<ExtractedRepeatPattern>& secondary_extracted_patterns);
    void update_extractor_cache(std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>& extracted_patterns,
                                const std::shared_ptr<ov::Model>& pattern,
                                const ov::NodeVector& pattern_node_vector,
                                std::map<std::string, InputInfo>& in_info);

};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
