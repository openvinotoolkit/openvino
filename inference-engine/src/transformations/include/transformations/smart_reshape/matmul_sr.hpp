// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReshapeAMatMul;
class TRANSFORMATIONS_API ReshapeBMatMul;
class TRANSFORMATIONS_API TransposeMatMul;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ReshapeAMatMul and ReshapeBMatMul transformations relax hard-coded Reshape followed by MatMul operation
 * For 2D Reshape search patterns are:
 *  - MatMul(Reshape(any_input, any_input), any_input)
 *  - MatMul(any_input, Reshape(any_input, any_input))
 */

class ov::pass::ReshapeAMatMul: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeAMatMul();
};
class ov::pass::ReshapeBMatMul: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeBMatMul();
};
class ov::pass::TransposeMatMul: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeMatMul();
};
