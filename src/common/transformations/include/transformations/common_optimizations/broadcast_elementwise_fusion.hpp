// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BroadcastElementwiseFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Removing Broadcast OP before ElementWise if output shape of Broadcast
 * are equal neighboring input shape of ElementWise.
 */

class ov::pass::BroadcastElementwiseFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastElementwiseFusion", "0");
    BroadcastElementwiseFusion();
};
