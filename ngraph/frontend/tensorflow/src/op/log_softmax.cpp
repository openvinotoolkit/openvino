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

OutputVector TranslateLogSoftmaxOp(const NodeContext& node) {
    auto ng_inp = node.get_ng_input(0);
    return {ConstructNgNode<LogSoftmax>(node.get_name(), ng_inp, -1)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
