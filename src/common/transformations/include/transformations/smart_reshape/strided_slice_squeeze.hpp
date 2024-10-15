// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StridedSliceSqueeze;
class TRANSFORMATIONS_API SqueezeStridedSlice;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief StridedSliceSqueeze transformation looks for SS -> Squeeze and corrects SS inputs and attributes for SS output
 * to be squeeze-able
 */

class ov::pass::StridedSliceSqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StridedSliceSqueeze", "0");
    StridedSliceSqueeze();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief StridedSliceSqueeze transformation looks for Squeeze -> SSe and corrects SS inputs and attributes for SS
 * output to be squeeze-able
 */

class ov::pass::SqueezeStridedSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SqueezeStridedSlice", "0");
    SqueezeStridedSlice();
};
