// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BroadcastTransition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief BroadcastTransition transformation moves broadcast through binary eltwise operation
 */
class ov::pass::BroadcastTransition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastTransition", "0");
    BroadcastTransition();
};
