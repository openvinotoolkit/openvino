// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
/**
 * @ingroup ov_transformation_common_api
 * @brief PATensorParallelFusion transformation matches following graph:
*/
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace ov {
namespace intel_gpu {

class PATensorParallelFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PATensorParallelFusion", "0");
    PATensorParallelFusion(size_t world_size, size_t world_rank);

private:
    std::shared_ptr<ov::Node> find_first_fc_after_pa(std::shared_ptr<ov::Node> input);
    std::shared_ptr<ov::Node> find_first_fc_before_pa(std::shared_ptr<ov::Node> input);
    void find_first_fcs_before_pa(std::shared_ptr<ov::Node> input);
    void find_ops_in_fc_to_pa(std::shared_ptr<ov::Node> input);
    void find_ops_in_pa_to_fc(std::shared_ptr<ov::Node> input);
    std::unordered_set<std::shared_ptr<ov::Node>> has_visited;
    std::vector<std::shared_ptr<ov::Node>> vector_visited;
    std::unordered_set<std::shared_ptr<ov::Node>> has_visited_fc;
    std::vector<std::shared_ptr<ov::Node>> vector_visited_fc;
};

}   // namespace intel_gpu
}   // namespace ov
