// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BroadcastConstRangeReplacement;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief BroadcastConstRangeReplacement replaces Constant filled with range values starting from 0 and replaces it with
 * Range op
 */

class ov::pass::BroadcastConstRangeReplacement : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastConstRangeReplacement", "0");
    BroadcastConstRangeReplacement();
};
