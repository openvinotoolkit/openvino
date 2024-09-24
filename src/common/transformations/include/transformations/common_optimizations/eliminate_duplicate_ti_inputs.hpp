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

class TRANSFORMATIONS_API EliminateDuplicateTIInputs;

}  // namespace pass
}  // namespace ov

/*
 * @ingroup ov_transformation_common_api
 * @brief EliminateDuplicateTIInputs transformation
 * removes duplicated inputs of SubgraphOps.
 */

class ov::pass::EliminateDuplicateTIInputs : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateDuplicateTIInputs", "0");
    EliminateDuplicateTIInputs();
};
