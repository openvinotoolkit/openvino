// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/extractimagepatches.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_extract_image_patches_op(const NodeContext& node) {
    default_op_checks(node, 1, {"ExtractImagePatches"});
    auto images = node.get_input(0);

    // retrieve attributes for ExtractImagePatches
    auto tf_ksizes = node.get_attribute<std::vector<int64_t>>("ksizes");
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_rates = node.get_attribute<std::vector<int64_t>>("rates");
    auto padding = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, padding);
    TENSORFLOW_OP_VALIDATION(node,
                             auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::VALID,
                             "[TensorFlow Frontend] Inconsistent model: only SAME and VALID padding modes are "
                             "supported for ExtractImagePatches.");

    // prepare attributes for opset ExtractImagePatches
    Shape sizes(2);
    Shape rates(2);
    Strides strides(2);
    convert_nhwc_to_hw(true, tf_ksizes, sizes);
    convert_nhwc_to_hw(true, tf_strides, strides);
    convert_nhwc_to_hw(true, tf_rates, rates);

    // prepare input to ExtractImagePatches
    convert_nhwc_to_nchw(true, images);

    Output<Node> extract_image_patches = make_shared<v3::ExtractImagePatches>(images, sizes, strides, rates, auto_pad);
    convert_nchw_to_nhwc(true, extract_image_patches);

    set_node_name(node.get_name(), extract_image_patches.get_node_shared_ptr());
    return {extract_image_patches};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
