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

OutputVector translate_range_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Range", "RANGE"});
    auto start = node.get_input(0);
    auto limit = node.get_input(1);
    auto delta = node.get_input(2);

    auto range = make_shared<Range>(start, limit, delta, start.get_element_type());
    set_node_name(node.get_name(), range);
    return {range};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
