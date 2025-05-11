// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_adjust_hue_op(const NodeContext& node) {
    default_op_checks(node, 2, {"AdjustHue"});
    auto images = node.get_input(0);
    auto delta = node.get_input(1);
    auto node_name = node.get_name();

    auto hsv_components = rgb_to_hsv(images.get_node_shared_ptr());
    auto hh = get<0>(*hsv_components);
    auto ss = get<1>(*hsv_components);
    auto vv = get<2>(*hsv_components);

    delta = make_shared<v1::ConvertLike>(delta, images);

    auto hh_adjust_ = make_shared<v1::Add>(hh, delta);
    auto hh_adjust_floor = make_shared<v0::Floor>(hh_adjust_);
    auto hh_adjust = make_shared<v1::Subtract>(hh_adjust_, hh_adjust_floor);

    auto new_images = hsv_to_rgb(hh_adjust, ss, vv);

    auto new_images_adjust_hue = new_images->output(0);

    set_node_name(node_name, new_images_adjust_hue.get_node_shared_ptr());
    return {new_images_adjust_hue};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
