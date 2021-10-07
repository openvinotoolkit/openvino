// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateMatMulOp(const NodeContext& node) {
    auto a = node.get_ng_input(0);
    auto b = node.get_ng_input(1);
    auto transpose_a = node.get_attribute<bool>("transpose_a", false);
    auto transpose_b = node.get_attribute<bool>("transpose_b", false);

    auto matmul = make_shared<MatMul>(a, b, transpose_a, transpose_b);
    matmul->set_friendly_name(node.get_name());
    return matmul->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
