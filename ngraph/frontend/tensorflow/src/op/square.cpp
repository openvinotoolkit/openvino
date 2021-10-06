// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

#if 0

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateSquareOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](Output<Node> n) {
        return ConstructNgNode<Multiply>(node.get_name(), n, n);
      });
}

}
}
#endif