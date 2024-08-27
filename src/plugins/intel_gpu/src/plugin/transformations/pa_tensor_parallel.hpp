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
    std::shared_ptr<ov::Node> fused_fc_before_pa(std::shared_ptr<ov::Node> input);
};

}   // namespace intel_gpu
}   // namespace ov
