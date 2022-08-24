// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class NGRAPH_API ReshapeSinkingMatMul;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ReshapeSinkingMatMul transformation looks for MatMul followed by optional Add
 * surrounded with Reshape operations which are only needed to merge and unmerge dimensions
 * into MatMuls batch. In case of success upscales MatMul to work with multidimensional batch and leaves single
 * Reshape operator after MatMul
 */

class ngraph::pass::ReshapeSinkingMatMul : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeSinkingMatMul", "0");
    ReshapeSinkingMatMul();
};
