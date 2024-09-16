// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief This transformation markups the marks paths that involve Range operations with floating point output data
 * types, as well as their users allowed for propagation. This pass is needed to prevent accuracy data loss in cases of
 * high range generation, which could suffer due to lowered precision.
 */
class TRANSFORMATIONS_API MarkFloatingPointRange : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkFloatingPointRange", "0");
    MarkFloatingPointRange();
};

OPENVINO_API void mark_range_path(const std::shared_ptr<Node>& node);
OPENVINO_API bool is_range_path(const std::shared_ptr<const Node>& node);
OPENVINO_API void erase_range_path(const std::shared_ptr<Node>& node);

}  // namespace pass
}  // namespace ov