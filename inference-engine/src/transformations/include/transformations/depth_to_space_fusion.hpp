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
    static bool check_depth_first(const ngraph::Shape& shape_input, const ngraph::Shape& shape_reshape_before,
                           const AxisVector& permutation, const ngraph::Shape& shape_reshape_after, size_t& possible_block_size);
    static bool check_block_first(const ngraph::Shape& shape_input, const ngraph::Shape& shape_reshape_before,
                           const AxisVector& permutation, const ngraph::Shape& shape_reshape_after, size_t& possible_block_size);
};
