// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConcatToBroadcast;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConcatToBroadcast transformation replaces Concat, having multiple inputs
 * from the same output, with a Broadcast node
 */
class ov::pass::ConcatToBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConcatToBroadcast", "0");
    ConcatToBroadcast();
};