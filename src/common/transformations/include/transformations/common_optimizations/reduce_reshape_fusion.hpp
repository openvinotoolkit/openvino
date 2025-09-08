// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReduceReshapeFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReduceReshapeFusion transformation
 * Fuse ReduceOp(keep_dims=false)+Reshape to ReduceOp(keep_dims=true)
 */
class ov::pass::ReduceReshapeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReduceReshapeFusion");
    ReduceReshapeFusion();
};
