// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReshapeTo1D;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ReshapeTo1D transformation looks for Reshape from nD to 1D tensor and replaces its pattern to [-1]
 */

class ov::pass::ReshapeTo1D : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeTo1D();
};
