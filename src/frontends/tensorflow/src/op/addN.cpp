// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_add_n_op(const NodeContext& node) {
    OutputVector ng_arg_vec = node.get_all_inputs();
    auto res = std::accumulate(std::next(ng_arg_vec.begin()),
                               ng_arg_vec.end(),
                               ng_arg_vec.at(0),
                               [](const Output<Node>& a, const Output<Node>& b) -> shared_ptr<Node> {
                                   return make_shared<Add>(a, b);
                               });
    set_node_name(node.get_name(), res.get_node_shared_ptr());
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov