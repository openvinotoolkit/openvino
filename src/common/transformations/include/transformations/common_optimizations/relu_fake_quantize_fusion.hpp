// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReluFakeQuantizeFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReluFakeQuantizeFusion transformation replaces following graph:
 * Relu -> FakeQuantize to FakeQuantize under following conditions:
 * -  'input_low' input to FakeQuantize is a Constant
 * -  'input_low' has non negative values
 */

class ov::pass::ReluFakeQuantizeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReluFakeQuantizeFusion", "0");
    ReluFakeQuantizeFusion();
};
