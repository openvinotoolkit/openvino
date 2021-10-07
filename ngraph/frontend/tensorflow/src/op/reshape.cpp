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

OutputVector TranslateReshapeOp(const NodeContext& node) {
    auto data = node.get_ng_input(0);
    auto shape = node.get_ng_input(1);
    auto reshape = make_shared<Reshape>(data, shape, false);
    reshape->set_friendly_name(node.get_name());
    return reshape->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
