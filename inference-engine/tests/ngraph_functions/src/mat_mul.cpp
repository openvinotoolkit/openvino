// Copyright (C) 2020 Intel Corporation
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

}  // namespace builder
}  // namespace ngraph