// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminateLoopInputsOutputs;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateLoopInputsOutputs transformation manages Loop inputs/outputs.
 * Check if Loop input is not changed in body
 * input node -> Loop input -> body parameter -> body result -> Loop output -> output node
 * than:
 * 1) reconnect input node -> output node directly
 * 2) update Loop input description from merged to invariant
 */

class ov::pass::EliminateLoopInputsOutputs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateLoopInputsOutputs");
    EliminateLoopInputsOutputs();
};
