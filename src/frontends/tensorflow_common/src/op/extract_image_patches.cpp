// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_extract_image_patches_op(const NodeContext& node) {
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() >= 0, "ExtractImagePatches must have at least one input.");
    auto images = node.get_input(0);

    // retrieve attributes for ExtractImagePatches
    auto tf_ksizes = node.get_attribute<std::vector<int64_t>>("ksizes");
    auto tf_strides = node.get_attribute<std::vector<int64_t>>("strides");
    auto tf_rates = node.get_attribute<std::vector<int64_t>>("rates");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    ov::op::PadType auto_pad = convert_tf_padding(node, tf_padding_type);
    TENSORFLOW_OP_VALIDATION(node,
                             auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::VALID,
                             "Only SAME_UPPER and VALID padding modes are supported for ExtractImagePatches.");

    // prepare attributes for OpenVINO ExtractImagePatches
    Shape sizes(2);
    Shape rates(2);
    Strides strides(2);

    // layout for this operation is always NHWC
    bool is_nhwc = true;
    convert_nhwc_to_hw(is_nhwc, tf_ksizes, sizes);
    convert_nhwc_to_hw(is_nhwc, tf_strides, strides);
    convert_nhwc_to_hw(is_nhwc, tf_rates, rates);

    // prepare input to ExtractImagePatches
    convert_nhwc_to_nchw(is_nhwc, images);

    auto extract_image_patches = make_shared<ExtractImagePatches>(images, sizes, strides, rates, auto_pad);

    // prepare output to return the original layout NHWC
    auto extract_image_patches_output = extract_image_patches->output(0);
    convert_nchw_to_nhwc(is_nhwc, extract_image_patches_output);

    set_node_name(node.get_name(), extract_image_patches_output.get_node_shared_ptr());
    return {extract_image_patches_output};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
