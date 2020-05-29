// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

    class INFERENCE_ENGINE_API_CLASS(DepthToSpaceFusion);

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *     DepthToSpaceFusion transformation detects Reshape-Transpose-Reshape pattern and
 *     tries to fuse it into a single DepthToSpace layer.
 *
 * Usage:
 *     DepthToSpaceFusion transformation is optional and disabled by default.
 *     The transformation can be enabled with callback using setCallback method.
 *     See the example below.
 *
 * Callback example:
 *
 *     // This callback enables DepthToSpaceFusion transformation
 *     auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
 *         return std::dynamic_pointer_cast<const ngraph::op::DepthToSpace>(node);
 *     };
 *
 *     auto p = ngraph::pass::DepthToSpaceFusion();
 *     p.setCallback(callback);
 *     p.run_on_function(f);
 *
 */

class ngraph::pass::DepthToSpaceFusion: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    DepthToSpaceFusion() : GraphRewrite(), PassParam() {
        depth_to_space_fusion();
    }

private:
    void depth_to_space_fusion();
};
