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

OutputVector TranslateXdivyOp(const NodeContext& node) {
    auto x = node.get_ng_input(0);
    auto y = node.get_ng_input(1);

    auto zero = make_shared<Constant>(x.get_element_type(), Shape{}, 0);
    auto one = make_shared<Constant>(x.get_element_type(), Shape{}, 1);
    auto x_is_zero = make_shared<Equal>(x, zero);

    // todo (itikhono) : looks wrong, verify
    // in OV TF it was:
    //    auto xdivy = make_shared<Divide>(x, y);
    //    auto select = make_shared<Select>(x_is_zero, y, xdivy);
    // current:
    auto select = make_shared<Select>(x_is_zero, one, y);
    auto xdivy = make_shared<Divide>(x, select);
    xdivy->set_friendly_name(node.get_name());
    return xdivy->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph