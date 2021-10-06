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
OutputVector TranslateSelectOp(const NodeContext& node) {
    TF_OP_VALIDATION_CHECK(node, node.get_all_ng_inputs().size() == 3, "Select op cannot be converted");
    auto in_1 = node.get_ng_input(0);
    auto in_2 = node.get_ng_input(1);
    auto in_3 = node.get_ng_input(2);
    if (in_1.get_partial_shape().is_static() && in_2.get_partial_shape().is_static()) {
        // select broadcast
        if (in_1.get_shape().size() == 1 and in_2.get_shape().size() > 1) {
            std::vector<uint64_t> axes(in_2.get_shape().size() - 1);
            std::iota(axes.begin(), axes.end(), 1);
            auto unsqueeze_axes = make_shared<Constant>(ngraph::element::i64, Shape{in_2.get_shape().size() - 1}, axes);
            auto unsqueeze = make_shared<Unsqueeze>(in_1, unsqueeze_axes);
            auto ng_select = make_shared<Select>(unsqueeze, in_2, in_3);
            ng_select->set_friendly_name(node.get_name());
            return ng_select->outputs();
        }
    }
    auto ng_select = make_shared<Select>(in_1, in_2, in_3);
    ng_select->set_friendly_name(node.get_name());
    return ng_select->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
