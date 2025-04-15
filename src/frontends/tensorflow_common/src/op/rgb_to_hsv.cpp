// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/concat.hpp"
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

    auto hsv_components = rgb_to_hsv(images);

    auto hh = get<0>(*hsv_components);
    auto ss = get<1>(*hsv_components);
    auto vv = get<2>(*hsv_components);

    auto rgb = make_shared<v0::Concat>(NodeVector{hh, ss, vv}, -1);

    set_node_name(node_name, rgb);
    return {rgb};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
