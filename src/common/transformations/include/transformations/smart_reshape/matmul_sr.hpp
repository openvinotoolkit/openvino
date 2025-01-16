// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReshapeAMatMul;
class TRANSFORMATIONS_API ReshapeBMatMul;
class TRANSFORMATIONS_API TransposeMatMul;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReshapeAMatMul and ReshapeBMatMul transformations relax hard-coded Reshape followed by MatMul operation
 * For 2D Reshape search patterns are:
 *  - MatMul(Reshape(any_input, any_input), any_input)
 *  - MatMul(any_input, Reshape(any_input, any_input))
 */

class ov::pass::ReshapeAMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapeAMatMul");
    ReshapeAMatMul();
};
class ov::pass::ReshapeBMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapeBMatMul");
    ReshapeBMatMul();
};
class ov::pass::TransposeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeMatMul");
    TransposeMatMul();
};
