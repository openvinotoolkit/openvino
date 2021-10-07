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

OutputVector TranslateWhereOp(const NodeContext& node) {
    auto x = node.get_ng_input(0);
    auto non_zero = make_shared<NonZero>(x);
    auto transpose_order = make_shared<Constant>(element::i64, Shape{2}, vector<int64_t>{1, 0});
    auto transpose = make_shared<opset8::Transpose>(non_zero, transpose_order);
    transpose->set_friendly_name(node.get_name());
    return transpose->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
