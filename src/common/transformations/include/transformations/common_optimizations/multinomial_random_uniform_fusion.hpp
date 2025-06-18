// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MultinomialRandomUniformFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief MultinomialRandomUniformFusion transformation replaces following graph:
 * Multinomial (no random_samples input) -> RandomUniform->Multinomial
 * Multinomial has the same params as the original, with added source of randomness
 */
class ov::pass::MultinomialRandomUniformFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MultinomialRandomUniformFusion");
    MultinomialRandomUniformFusion();
};
