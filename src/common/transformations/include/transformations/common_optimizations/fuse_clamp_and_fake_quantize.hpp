// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FuseClampAndFakeQuantize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief FuseClampAndFakeQuantize removes Clamp before FakeQuantize when the Clamp interval fully covers the
 * FakeQuantize input interval and Clamp has a single consumer.
 *
 * The transformation requires FakeQuantize input_low and input_high to be Constant nodes.
 */
class ov::pass::FuseClampAndFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseClampAndFakeQuantize");
    FuseClampAndFakeQuantize();
};