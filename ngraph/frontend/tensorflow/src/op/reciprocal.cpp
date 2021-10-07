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

OutputVector TranslateReciprocalOp(const NodeContext& node) {
    auto x = node.get_ng_input(0);
    auto ng_exponent = make_shared<Constant>(x.get_element_type(), Shape{}, -1);
    auto power = make_shared<Power>(x, ng_exponent);
    power->set_friendly_name(node.get_name());
    return power->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
