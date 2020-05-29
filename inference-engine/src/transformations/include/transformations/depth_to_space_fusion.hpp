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

class ngraph::pass::DepthToSpaceFusion: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    DepthToSpaceFusion() : GraphRewrite(), PassParam() {
        depth_to_space_fusion();
    }

private:
    void depth_to_space_fusion();
};
