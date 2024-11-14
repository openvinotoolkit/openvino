// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets_helpers.hpp"
#include "snippets/utils/utils.hpp"

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
            OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
            OPENVINO_ASSERT(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
            OPENVINO_ASSERT(ov::op::util::is_binary_elementwise_arithmetic(customOps[0]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[1]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[2]),
                         "Got invalid custom ops: expected binary and two unary operations");
            OPENVINO_ASSERT(input_shapes[0].is_static() && input_shapes[1].is_static(), "This test supports only static shapes");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

/// Convolution followed by a Bias and Relu
/// Tokenization will be skipped, if defined OV_CPU_WITH_ACL. Because Bias and Relu will be fused as post-ops.
//     in1           in2
// Convolution      Const
//             Bias
//             Relu
//            Result
class ConvBiasActivationFunction : public SnippetsFunctionCustomizable {
public:
    explicit ConvBiasActivationFunction(const std::vector<PartialShape>& inputShapes, const std::vector<std::shared_ptr<Node>>& customOps)
            : SnippetsFunctionCustomizable(inputShapes, customOps, {1, 1}) {
            OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
            OPENVINO_ASSERT(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
            OPENVINO_ASSERT(ov::op::util::is_binary_elementwise_arithmetic(customOps[0]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[1]),
                         "Got invalid custom ops: expected a Bias and unary operation");
            OPENVINO_ASSERT(input_shapes[0].is_static(), "This test supports only static shapes");
            OPENVINO_ASSERT(ov::is_type<ov::op::v1::Add>(customOps[0]), "Need an Add node to be Bias");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/// Convolution followed by a Bias and two Relus
/// The second Relu will be tokenized, if defined OV_CPU_WITH_ACL. Because only Bias and the first Relu will be fused as post-ops.
//     in1           in2
// Convolution      Const
//             Bias
//             Relu
//             Relu
//            Result
class ConvBiasTwoActivationFunction : public SnippetsFunctionCustomizable {
public:
    explicit ConvBiasTwoActivationFunction(const std::vector<PartialShape>& inputShapes, const std::vector<std::shared_ptr<Node>>& customOps)
            : SnippetsFunctionCustomizable(inputShapes, customOps, {1, 1, 1}) {
            OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
            OPENVINO_ASSERT(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
            OPENVINO_ASSERT(ov::op::util::is_binary_elementwise_arithmetic(customOps[0]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[1]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[2]),
                         "Got invalid custom ops: expected a Bias and two unary operations");
            OPENVINO_ASSERT(input_shapes[0].is_static(), "This test supports only static shapes");
            OPENVINO_ASSERT(ov::is_type<ov::op::v1::Add>(customOps[0]), "Need an Add node to be Bias");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

/// MatMul followed by a Bias and two Relus
/// Not tokenizable, because Bias and two Relus will all be fused as post-ops.
//     in1          in2
//    MatMul       Const
//           Bias
//           Relu
//           Relu
//          Result
class MatMulTwoActivationFunction : public SnippetsFunctionCustomizable {
public:
    explicit MatMulTwoActivationFunction(const std::vector<PartialShape>& inputShapes, const std::vector<std::shared_ptr<Node>>& customOps)
            : SnippetsFunctionCustomizable(inputShapes, customOps, {2, 1, 1}) {
            OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
            OPENVINO_ASSERT(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
            OPENVINO_ASSERT(ov::op::util::is_binary_elementwise_arithmetic(customOps[0]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[1]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[2]),
                         "Got invalid custom ops: expected a Bias and two unary operations");
            OPENVINO_ASSERT(input_shapes[0].is_static() && input_shapes[1].is_static(), "This test supports only static shapes");
            OPENVINO_ASSERT(ov::is_type<ov::op::v1::Add>(customOps[0]), "Need an Add node to be Bias");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

/// MatMul followed by a Bias, Relu and Div
/// The Div will be tokenized, if defined OV_CPU_WITH_ACL. Because only Bias and the first Relu will be fused as post-ops.
//     in1          in2
//    MatMul       Const
//           Bias
//           Relu
//           Div
//          Result
class MatMulBiasActivationBinaryFunction : public SnippetsFunctionCustomizable {
public:
    explicit MatMulBiasActivationBinaryFunction(const std::vector<PartialShape>& inputShapes, const std::vector<std::shared_ptr<Node>>& customOps)
            : SnippetsFunctionCustomizable(inputShapes, customOps, {2, 1, 2}) {
            OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
            OPENVINO_ASSERT(input_shapes[0].size() == 4, "Only 4D input shapes are currently supported");
            OPENVINO_ASSERT(ov::op::util::is_binary_elementwise_arithmetic(customOps[0]) &&
                         ov::op::util::is_unary_elementwise_arithmetic(customOps[1]) &&
                         ov::op::util::is_binary_elementwise_arithmetic(customOps[2]),
                         "Got invalid custom ops: expected a Bias , unray, and binary operation");
            OPENVINO_ASSERT(input_shapes[0].is_static() && input_shapes[1].is_static() && input_shapes[2].is_static(), "This test supports only static shapes");
            OPENVINO_ASSERT(ov::is_type<ov::op::v1::Add>(customOps[0]), "Need an Add node to be Bias");
    }
private:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
