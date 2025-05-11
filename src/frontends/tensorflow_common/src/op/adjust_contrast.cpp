// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_adjust_contrast_op(const NodeContext& node) {
    default_op_checks(node, 2, {"AdjustContrastv2"});
    auto images = node.get_input(0);
    auto contrast_factor = node.get_input(1);
    auto node_name = node.get_name();

    // compute mean per channel for each image
    // it will reduce spatial dimensions of images in a format [batch, height, width, channel]
    auto reduce_axes = make_shared<v0::Constant>(element::i32, Shape{2}, vector<int32_t>{-3, -2});
    auto means = make_shared<v1::ReduceMean>(images, reduce_axes, true);

    // cast contrast_factor since its type can be different
    contrast_factor = make_shared<v1::ConvertLike>(contrast_factor, images);

    // adjust contrast by a formula: (images - means) * contrast_factor + means
    auto adjust_contrast = make_shared<v1::Subtract>(images, means)->output(0);
    adjust_contrast = make_shared<v1::Multiply>(adjust_contrast, contrast_factor);
    adjust_contrast = make_shared<v1::Add>(adjust_contrast, means);

    set_node_name(node_name, adjust_contrast.get_node_shared_ptr());

    return {adjust_contrast};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
