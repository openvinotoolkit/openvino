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

OutputVector TranslateTransposeOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto order = node.get_ng_input(1);
    auto transpose = make_shared<opset8::Transpose>(input, order);
    transpose->set_friendly_name(node.get_name());
    return transpose->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph