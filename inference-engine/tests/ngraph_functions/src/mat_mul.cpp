// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeMatMul(const Output<Node>& A,
                                 const Output<Node>& B) {
    return std::make_shared<ngraph::opset3::MatMul>(A, B);
}

}  // namespace builder
}  // namespace ngraph