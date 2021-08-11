// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

    class TRANSFORMATIONS_API ShuffleChannelsFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ShuffleChannelsFusion transformation detects Reshape-Transpose-Reshape pattern
 * and tries to fuse it into a single ShuffleChannels layer with axis = 1.
 *
 * x'  = reshape(x, [N, group, C / group, H, W]) or reshape(x, [N, group, C / group, H * W])
 * x'' = transpose(x', [0, 2, 1, 3, 4]) or transpose(x', [0, 2, 1, 3])
 * y   = reshape(x'', [N, C, H, W])
 *
 * @param reshape_constants_check the flag that defines the need for additional checks of reshapes constant
 *        Additional checks are required when ShuffleChannelsFusion using inside offline transformations
 *        and are not necessary when ShuffleChannelsFusion using inside CommonOptimizations
 */

class ov::pass::ShuffleChannelsFusion : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ShuffleChannelsFusion(const bool reshape_constants_check);
};
