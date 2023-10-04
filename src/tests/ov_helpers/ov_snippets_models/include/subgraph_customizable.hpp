// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "snippets_helpers.hpp"

/* This file contains definitions of rather complex functions (models) that support (and require)
 * specification of some the internal operations. This flexibility is required to extend coverage of
 * different tokenization scenarios in parametrized tests. All the functions are expected to be direct
 * descendants of SnippetsFunctionCustomizable (defined in snippets_helpers.hpp).
 */

namespace ov {
namespace test {
namespace snippets {
// Todo: Remove Sinh, when Subgraph stop skipping eltwises (Transposes and Converts) after inputs

/// Convolution followed by a two-input Multiply, Relu and Sqrt
/// Tokenized by attaching eltwises, but becomes non-tokenizable if Multiply is substituted with Add (CPU-specific fusing)
//    in1          in2
// Convolution     Sinh
//         Multiply
//           Relu
//           Sqrt
//          Result
class ConvMulActivationFunction : public SnippetsFunctionCustomizable {
public:
    explicit ConvMulActivationFunction(const std::vector<PartialShape>& inputShapes, const std::vector<std::shared_ptr<Node>>& customOps)
            : SnippetsFunctionCustomizable(inputShapes, customOps, {2, 1, 1}) {
            NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
            NGRAPH_CHECK(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
            NGRAPH_CHECK(ov::op::util::is_binary_elementwise_arithmetic(customOps[0]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[1]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[2]),
                         "Got invalid custom ops: expected binary and two unary operations");
            NGRAPH_CHECK(input_shapes[0].is_static() && input_shapes[1].is_static(), "This test supports only static shapes");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
