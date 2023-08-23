// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API InterpolateSequenceFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief InterpolateSequenceFusion transformation replaces a sequence of
 *        operations to Interpolate op.
 */
class ov::pass::InterpolateSequenceFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InterpolateSequenceFusion", "0");
    InterpolateSequenceFusion();
};
