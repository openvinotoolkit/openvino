// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_rgb_to_hsv_op(const NodeContext& node) {
    default_op_checks(node, 1, {"RGBToHSV"});
    auto images = node.get_input(0);
    auto node_name = node.get_name();

    auto new_images = rgb_to_hsv(images);

    set_node_name(node_name, new_images);
    return {new_images};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov