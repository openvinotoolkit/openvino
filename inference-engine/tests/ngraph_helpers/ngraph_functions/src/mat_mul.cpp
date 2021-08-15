// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeMatMul(const Output<Node>& A,
                                 const Output<Node>& B,
                                 bool transpose_a,
                                 bool transpose_b) {
    return std::make_shared<ngraph::opset3::MatMul>(A, B, transpose_a, transpose_b);
}

std::shared_ptr<Node> makeMatMulRelaxed(const Output<Node>& A,
                                        const Output<Node>& B,
                                        const element::Type &elemType,
                                        bool transpose_a,
                                        bool transpose_b) {
    auto inputParamsFP32 = ngraph::builder::makeParams(ngraph::element::f32, { A.get_shape() });
    auto matrixB_FP32 = ngraph::builder::makeConstant<float>(ngraph::element::f32, B.get_shape(), {}, true);

    // todo: fill values

    auto matMulNodeRelaxed = std::make_shared<op::TypeRelaxed<opset1::MatMul>>(
            *as_type_ptr<opset1::MatMul>(std::make_shared<ngraph::opset1::MatMul>(inputParamsFP32[0], matrixB_FP32, transpose_a, transpose_b)),
                    element::f32);

    auto newMatrixB = ngraph::builder::makeConstant<float>(elemType, B.get_shape(), {}, true, 5.3f, -5.3f);

    auto newMatMul = matMulNodeRelaxed->copy_with_new_inputs({A, newMatrixB});

    return newMatMul;
}

}  // namespace builder
}  // namespace ngraph