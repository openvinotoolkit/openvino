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
public:
    using InputVector = std::vector<ov::Input<ov::Node>>;
    using OutputVector = std::vector<ov::Output<ov::Node>>;

    std::list<ExtractedPattern> extract(const std::shared_ptr<ov::Model> &model,
                                        bool is_extract_body = true,
                                        bool is_copy_constants = true) override;

    std::vector<std::vector<std::pair<InputVector, OutputVector>>>
    get_repeat_pattern_borders(const std::shared_ptr<ov::Model> &model,
                                bool is_extract_body = true,
                                bool is_recursive_extraction = true,
                                bool is_copy_constants = true);

    std::vector<std::vector<ov::NodeVector>>
    get_repeat_node_vectors(const std::shared_ptr<ov::Model> &model,
                            bool is_extract_body = true,
                            bool is_recursive_extraction = true,
                            bool is_copy_constants = true);

private:
    using ExtractedRepeatPattern = std::pair<std::map<std::shared_ptr<ov::Model>, ov::NodeVector>, std::map<std::string, InputInfo>>;
    ModelComparator::Ptr model_comparator = ModelComparator::get();

    // find patterns in original models
    // { pattern, in_info, { NodeVector }}
    std::vector<ExtractedRepeatPattern>
    find_repeat_patterns(const std::shared_ptr<ov::Model> &model,
                         bool is_extract_body = true,
                         bool is_recursive_extraction = true,
                         bool is_copy_constants = true,
                         bool is_borders = false);
    void update_extractor_cache(std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>& extracted_patterns,
                                std::vector<ExtractedRepeatPattern>& secondary_extracted_patterns);
    void update_extractor_cache(std::vector<RepeatPatternExtractor::ExtractedRepeatPattern>& extracted_patterns,
                                const std::shared_ptr<ov::Model>& model,
                                const ov::NodeVector& node_vector,
                                std::map<std::string, InputInfo>& in_info);

};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
