// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BinarizeWeights;

}  // namespace pass
}  // namespace ov

// clang-format off
/**
 * @ingroup ov_transformation_common_api
 * @brief This transformation converts weights to -1/+1 form
 * and applies normalization factors to output low/high and after Convolution.
 * For example, following graph
 *
 *         .... ....  out_low  out_high           weights ..    ..  out_low out_high
 *           |    |      |        |                  |     |    |      |     |
 *          +--------------------------+           +--------------------------+
 *          | FakeQuantize (levels==2) |           | FakeQuantize (levels==2) |
 *          |     (on activations)     |           |       (on weights)       |
 *          +--------------------------+           +--------------------------+
 *                        |                                      |
 *                        |                                      |
 *                        -----------------    -------------------
 *                                        |    |
 *                                        v    v
 *                                   +-------------+
 *                                   | Convolution |
 *                                   +-------------+
 *                                          |
 *                                          v
 *
 * is transformed to:
 *
 *                  normalized normalized
 *         .... ....  out_low   out_high
 *           |    |      |         |
 *          +--------------------------+           +--------------------------+
 *          | FakeQuantize (levels==2) |           |         Constant         |
 *          |     (on activations)     |           | (with converted weights) |
 *          +--------------------------+           +--------------------------+
 *                        |                                      |
 *                        |                                      |
 *                        -----------------    -------------------
 *                                        |    |
 *                                        v    v
 *                                   +-------------+
 *                                   | Convolution |
 *                                   +-------------+
 *                                          |
 *                                          v
 *                                   +------------+     +---------------------------------------------------------------+
 *                                   |  Multiply  | <---| Constant (normalization factor coming from FQ on activations) |
 *                                   +------------+     +---------------------------------------------------------------+
 *                                          |
 *                                          v
 *                                   +------------+     +-----------------------------------------------------------+
 *                                   |  Multiply  | <---| Constant (normalization factor coming from FQ on weights) |
 *                                   +------------+     +------------------------------------------------------------
 *                                          |
 *                                          v
 *
 * Normalization factors are chosen based output_high value.
 * If it's zero - norm factor is equal to output_low and output_high otherwise
 */
// clang-format on

class ov::pass::BinarizeWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BinarizeWeights");
    BinarizeWeights();
};
