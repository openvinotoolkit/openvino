// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DepthToSpaceFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief DepthToSpaceFusion transformation detects Reshape-Transpose-Reshape pattern
 * and tries to fuse it into a single DepthToSpace layer.
 *
 * DepthToSpaceFusion transformation is optional and disabled by default.
 * The transformation can be enabled with callback using setCallback method.
 * See the example below.
 *
 * Callback example:
 *
 *     // This callback enables DepthToSpaceFusion transformation
 *     auto callback = [](const std::shared_ptr<const ov::Node> & node) -> bool {
 *         return std::dynamic_pointer_cast<const ov::opset3::DepthToSpace>(node) != nullptr;
 *     };
 *
 *     auto p = ov::pass::DepthToSpaceFusion();
 *     p.setCallback(callback);
 *     p.run_on_function(f);
 *
 */

class ov::pass::DepthToSpaceFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DepthToSpaceFusion", "0");
    DepthToSpaceFusion();
};
