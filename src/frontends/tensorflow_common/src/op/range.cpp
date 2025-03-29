// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"

#include "common_op_table.hpp"
#include "openvino/op/convert_like.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_range_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Range", "RANGE"});
    auto start = node.get_input(0);
    auto limit = node.get_input(1);
    auto delta = node.get_input(2);

    auto start_type = start.get_element_type();
    Output<Node> range;
    if (start_type.is_static()) {
        range = make_shared<v4::Range>(start, limit, delta, start_type);
    } else {
        range = make_shared<v4::Range>(start, limit, delta, element::f32);
        range = make_shared<v1::ConvertLike>(range, start);
    }
    set_node_name(node.get_name(), range.get_node_shared_ptr());
    return {range};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
