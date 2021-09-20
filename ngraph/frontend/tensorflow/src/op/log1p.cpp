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

OutputVector TranslateLog1pOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](Output<Node> n) {
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> val_1(shape_size(shape), "1");
        auto ng_const1 =
            ConstructNgNode<opset::Constant>(node.get_name(), et, shape, val_1);
        auto ng_add = ConstructNgNode<opset::Add>(node.get_name(), ng_const1, n);
        return ConstructNgNode<opset::Log>(node.get_name(), ng_add);
      });
}
}
}
#endif