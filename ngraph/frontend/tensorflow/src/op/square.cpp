// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateSquareOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](Output<Node> n) {
        return ConstructNgNode<opset::Multiply>(node.get_name(), n, n);
      });
}

}
}
#endif