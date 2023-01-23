// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_clip_by_value_op(const NodeContext& node) {
    default_op_checks(node, 3, {"ClipByValue"});
    auto t = node.get_input(0);
    auto clip_value_min = node.get_input(1);
    auto clip_value_max = node.get_input(2);

    // it can be case that clip_value_min > clip_value_max
    // in this case both values are equal to clip_value_min
    clip_value_max = make_shared<Maximum>(clip_value_min, clip_value_max);

    auto clip_by_min = make_shared<Maximum>(t, clip_value_min);
    auto clip_by_max = make_shared<Minimum>(clip_by_min, clip_value_max);

    set_node_name(node.get_name(), clip_by_max);
    return {clip_by_max};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
