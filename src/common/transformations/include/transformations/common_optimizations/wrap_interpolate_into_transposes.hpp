// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API WrapInterpolateIntoTransposes;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief WrapInterpolateIntoTransposes transformation replaces
 *      Interpolate
 *  with
 *      Transpose -> Interpolate -> Transpose
 *  when
 *    1) the source Interpolate has the static input rank;
 *    2) 'axes' input is a Constant;
 *    3) number of axes is equal to input rank minus 2;
 *    4) axes contain 0 or 1.
 *  The reason of this transformation is that now CPU plugin supports interpolation only
 *  with respect to spatial dimensions, but TensorFlow frontend gives Interpolate with
 *  axes {1, 2} for 4D tensors.
 */
class ngraph::pass::WrapInterpolateIntoTransposes : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("WrapInterpolateIntoTransposes", "0");
    WrapInterpolateIntoTransposes();
};
