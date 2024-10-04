// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvToBinaryConv;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief This transformation converts Convolution to BinaryConvolution under following conditions:
 *  - first input to Convolution is a FakeQuantize with levels==2 with output low,high being either (0, 1) or (-1, 1)
 *  - second input (weights) is a constant with values -1 or 1
 * The transformation also converts weights to binary Constant (with 'u1' type)
 * For example, when output_low is equal to 0 and output_high is equal to 1, following graph
 *
 *         .... ....  out_low   out_high
 *           |    |      |         |
 *          +--------------------------+           +-------------------------------------+
 *          | FakeQuantize (levels==2) |           |               Constant              |
 *          |     (on activations)     |           | (weights containing -1 or 1 values) |
 *          +--------------------------+           +-------------------------------------+
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
 * is transformed to:
 *
 *         .... ....  out_low   out_high
 *           |    |      |         |
 *          +--------------------------+           +---------------------------------+
 *          | FakeQuantize (levels==2) |           |     Constant (with u1 type)     |
 *          |     (on activations)     |           | (with u1 type - binary weights) |
 *          +--------------------------+           +---------------------------------+
 *                        |                                      |
 *                        |                                      |
 *                        -----------------    -------------------
 *                                        |    |
 *                                        v    v
 *                                +-------------------+
 *                                | BinaryConvolution |
 *                                +-------------------+
 *                                          |
 *                                          v
 *                                   +------------+     +----------------------------------------------------+
 *                                   |            |     |                   Constant                         |
 *                                   |     Add    | <---|          (weights from original graph,             |
 *                                   |            |     |  sum-reduced over [1,..., len(weights.shape)] axes |
 *                                   +------------+     +----------------------------------------------------+
 *                                          |
 *                                          v
 *                                   +------------+     +-----+
 *                                   |  Multiply  | <---| 0.5 |
 *                                   +------------+     +-----+
 *                                          |
 *                                          v
 */
class ov::pass::ConvToBinaryConv : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvToBinaryConv", "0");
    ConvToBinaryConv();
};
