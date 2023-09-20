// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeMatMul(const Output<Node>& A, const Output<Node>& B, bool transpose_a, bool transpose_b) {
    return std::make_shared<ov::op::v0::MatMul>(A, B, transpose_a, transpose_b);
}

}  // namespace builder
}  // namespace ngraph
