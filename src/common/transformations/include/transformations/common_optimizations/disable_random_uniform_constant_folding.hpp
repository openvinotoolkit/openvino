// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DisableRandomUniformConstantFolding;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Disables ConstantFolding for RandomUniform operation. It is required as RandomUniform
 * should generate new sequence each run.
 */
class ov::pass::DisableRandomUniformConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableRandomUniformConstantFolding");
    DisableRandomUniformConstantFolding();
};
