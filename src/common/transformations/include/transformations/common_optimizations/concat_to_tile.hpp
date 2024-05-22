// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConcatToTile;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConcatToTile transformation replaces Concat, having multiple inputs
 * from the same output, with a Broadcast node
 */
class ov::pass::ConcatToTile : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConcatToTile", "0");
    ConcatToTile();
};