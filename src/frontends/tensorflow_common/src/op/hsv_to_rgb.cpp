// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_hsv_to_rgb_op(const NodeContext& node) {
    default_op_checks(node, 1, {"HSVToRGB"});
    auto images = node.get_input(0);
    auto node_name = node.get_name();

    auto const_minus_one_i = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto channels = make_shared<v1::Split>(images, const_minus_one_i, 3);

    auto hh = channels->output(0);
    auto ss = channels->output(1);
    auto vv = channels->output(2);

    auto new_images = hsv_to_rgb(hh, ss, vv);

    set_node_name(node_name, new_images);
    return {new_images};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
