// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_mlp.hpp"
#include "openvino/opsets/opset15.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> MLPSeqFunction::initOriginal() const {
    auto A = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto B = std::make_shared<ov::op::v0::Parameter>(precisions[1], input_shapes[1]);
    auto add = std::make_shared<ov::op::v0::Parameter>(precisions[2], input_shapes[2]);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(A, B);

    std::shared_ptr<Node> current = matmul;

    for (size_t i = 0; i < num_layers; ++i) {
        current = std::make_shared<ov::op::v1::Add>(matmul, add);
    }

    auto result = std::make_shared<ov::op::v0::Result>(current);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A, B, add});
}

std::shared_ptr<ov::Model> MLPSeqFunction::initReference() const {
    // Reference implementation can be added here if needed
    return initOriginal();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
