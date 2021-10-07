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

OutputVector TranslateLog1pOp(const NodeContext& node) {
    auto n = node.get_ng_input(0);
    auto const_1 = make_shared<Constant>(n.get_element_type(), Shape{}, 1);
    auto add = make_shared<Add>(n, const_1);
    auto log = make_shared<Log>(add);
    log->set_friendly_name(node.get_name());
    return log->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
