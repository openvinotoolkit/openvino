// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_adjust_saturation_op(const NodeContext& node) {
    default_op_checks(node, 2, {"AdjustSaturation"});
    auto images = node.get_input(0);
    auto scale = node.get_input(1);
    auto node_name = node.get_name();

    auto hsv_components = rgb_to_hsv(images.get_node_shared_ptr());

    auto hh = get<0>(*hsv_components);
    auto ss = get<1>(*hsv_components);
    auto vv = get<2>(*hsv_components);

    scale = make_shared<v1::ConvertLike>(scale, images);

    auto ss_adjust = make_shared<v0::Clamp>(make_shared<v1::Multiply>(ss, scale), 0.0f, 1.0f);

    auto new_images = hsv_to_rgb(hh, ss_adjust, vv);

    auto new_images_adjust_saturation = new_images->output(0);

    set_node_name(node_name, new_images_adjust_saturation.get_node_shared_ptr());
    return {new_images_adjust_saturation};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
