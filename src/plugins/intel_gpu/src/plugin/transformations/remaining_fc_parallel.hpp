// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
/*
 * @ingroup ov_transformation_common_api
 * @brief RemainFCParallelFusion transformation matches following graph:
*/
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace ov {
namespace intel_gpu {

class RemainFCParallelFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemainFCParallelFusion", "0");
    RemainFCParallelFusion(size_t world_size, size_t world_rank);
    std::shared_ptr<ov::Node> find_first_fc_after_multiply(std::shared_ptr<ov::Node> root_node);
};

}   // namespace intel_gpu
}   // namespace ov
