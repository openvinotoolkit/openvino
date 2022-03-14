// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DivisionByZeroFP16Resolver;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief: clamps eps into fp16 minimal normalized value in input_1/Maximum(input_2, eps); input_1/Add(input_2, eps);
 * and input_1*Pow(Maximum[Add](input_2, eps), -z) patterns to prevent division by zero.
 *
 * eps must be always nonzero to prevent from NaNs in such expressions if input_1 and input_2 simultaneously happened to
 * be zero. We should keep in such patterns eps >= fp16 minimal normalized value so that CompressFloatConstants should
 * not cast them into zero during compression into f16.
 */
class ov::pass::DivisionByZeroFP16Resolver : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DivisionByZeroFP16Resolver", "0");
    DivisionByZeroFP16Resolver();
};
