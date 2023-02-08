// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_linspace_op(const NodeContext& node) {
    // The operation is simple that generates a range [start, ..., stop]
    // with num elements staying in the same distance between each other
    default_op_checks(node, 3, {"LinSpace"});
    auto start = node.get_input(0);
    auto stop = node.get_input(1);
    auto num = node.get_input(2);

    // compute delta value, i.e. distance between neighbor values of the result
    auto const_one = make_shared<Constant>(num.get_element_type(), Shape{}, 1);
    Output<Node> num_minus_one = make_shared<Subtract>(num, const_one);
    num_minus_one = make_shared<Convert>(num_minus_one, start.get_element_type());
    Output<Node> delta = make_shared<Subtract>(stop, start);
    delta = make_shared<Divide>(delta, num_minus_one);

    // compute the result
    auto stop_plus_delta = make_shared<Add>(stop, delta);
    auto linspace = make_shared<Range>(start, stop_plus_delta, delta, start.get_element_type());
    set_node_name(node.get_name(), linspace);
    return {linspace};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
