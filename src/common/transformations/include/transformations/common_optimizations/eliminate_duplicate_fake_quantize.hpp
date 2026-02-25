// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminateDuplicateFakeQuantize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateDuplicateFakeQuantize transformation eliminates redundant
 * cascaded FakeQuantize operations: FQ1 -> FQ2 where both operate with same levels.
 * 
 * Pattern: Input -> FakeQuantize1 -> FakeQuantize2 -> Output
 * 
 * The transformation merges two consecutive FakeQuantize layers into a single one when:
 * - Both FakeQuantize operations have the same number of levels
 * - FQ1's output range is equivalent to FQ2's input range
 * 
 * Result: Input -> FakeQuantize_merged -> Output
 * where merged FQ uses FQ1's input range and FQ2's output range.
 * 
 * This optimization reduces computation overhead and improves inference performance
 * by eliminating unnecessary quantization-dequantization cycles.
 */
class ov::pass::EliminateDuplicateFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateDuplicateFakeQuantize");
    EliminateDuplicateFakeQuantize();
};
