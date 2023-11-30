// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ReshapeOptimizations;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Searches for Flatten-like Reshape operations and simplifies 2nd input of such Reshape using special zero
 * feature
 */
class ov::pass::ReshapeOptimizations : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeOptimizations", "0");
    ReshapeOptimizations();
};
