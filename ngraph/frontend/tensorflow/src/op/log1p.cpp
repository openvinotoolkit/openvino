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

OutputVector TranslateLog1pOp(
    const NodeContext& node) {
  return TranslateUnaryOp(
      op, static_input_map, ng_op_map, [&op](Output<Node> n) {
        auto et = n.get_element_type();
        auto shape = n.get_shape();
        std::vector<std::string> val_1(shape_size(shape), "1");
        auto ng_const1 =
            ConstructNgNode<Constant>(node.get_name(), et, shape, val_1);
        auto ng_add = ConstructNgNode<Add>(node.get_name(), ng_const1, n);
        return ConstructNgNode<Log>(node.get_name(), ng_add);
      });
}
}
}
#endif