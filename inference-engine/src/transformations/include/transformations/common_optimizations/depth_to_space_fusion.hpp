// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API DepthToSpaceFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
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
 *     auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
 *         return std::dynamic_pointer_cast<const ngraph::opset3::DepthToSpace>(node) != nullptr;
 *     };
 *
 *     auto p = ngraph::pass::DepthToSpaceFusion();
 *     p.setCallback(callback);
 *     p.run_on_function(f);
 *
 */

class ngraph::pass::DepthToSpaceFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DepthToSpaceFusion();
};
