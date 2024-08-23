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

class TRANSFORMATIONS_API SoftPlusToMishFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SoftPlusToMishFusion transformation replaces group of
 * operations: x * tanh(softplus(x)) to Mish op.
 */
class ov::pass::SoftPlusToMishFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftPlusToMishFusion", "0");
    SoftPlusToMishFusion();
};
