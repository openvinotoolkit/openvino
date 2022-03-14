// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertInterpolate1ToInterpolate4;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertInterpolate1ToInterpolate4 covert v0:interpolate into v4::Interpolate.
 */
class ngraph::pass::ConvertInterpolate1ToInterpolate4 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertInterpolate1ToInterpolate4", "0");
    ConvertInterpolate1ToInterpolate4();
};
