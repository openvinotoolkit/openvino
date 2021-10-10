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

OutputVector TranslateSqueezeOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto axes = node.get_attribute<std::vector<int32_t>>("squeeze_dims");
    auto axes_const = make_shared<Constant>(element::i32, Shape{axes.size()}, axes);
    auto squeeze = make_shared<Squeeze>(input, axes_const);
    squeeze->set_friendly_name(node.get_name());
    return squeeze->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph