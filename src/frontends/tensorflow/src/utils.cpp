// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

void ov::frontend::tensorflow::tf_shape_to_ov_shape(const ::tensorflow::TensorShapeProto& tf_shape,
                                                    ov::PartialShape* ng_shape) {
    std::vector<ov::Dimension> dims;
    for (int i = 0; i < tf_shape.dim_size(); i++) {
        dims.emplace_back(tf_shape.dim(i).size());
    }
    *ng_shape = ov::PartialShape(dims);
}

void ov::frontend::tensorflow::set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node) {
    const auto& outputs = node->outputs();
    node->set_friendly_name(node_name);
    if (outputs.size() == 1) {
        set_out_name(node_name, outputs[0]);
    }
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        set_out_name({node_name + ":" + std::to_string(idx)}, outputs[idx]);
    }
}

void ov::frontend::tensorflow::set_out_name(const std::string& out_name, const ov::Output<ov::Node>& output) {
    output.get_tensor().add_names({out_name});
}

ov::op::PadType ov::frontend::tensorflow::convert_conv_tf_padding(const ov::frontend::tensorflow::NodeContext& node,
                                                                  const std::string& tf_padding) {
    auto op_type = node.get_op_type();

    TENSORFLOW_OP_VALIDATION(node,
                             op_type == "Conv2D" || op_type == "Conv2DBackpropInput" || op_type == "Conv3D" ||
                                 op_type == "Conv3DBackpropInputV2",
                             "The convert_conv_tf_padding routine supports only convolutional operations.");
    TENSORFLOW_OP_VALIDATION(
        node,
        tf_padding == "VALID" || tf_padding == "SAME" || tf_padding == "EXPLICIT",
        "The deconvolutional operation must have one of the padding type: VALID, SAME, and EXPLICIT.");

    if (tf_padding == "VALID") {
        return ov::op::PadType::VALID;
    }
    if (node.get_op_type() == "Conv2DBackpropInput" || node.get_op_type() == "Conv3DBackpropInputV2") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // ConvBackpropData layer in the Operation specification,
            // the SAME_LOWER value matches to the SAME value in TensorFlow
            return ov::op::PadType::SAME_LOWER;
        }
    } else if (node.get_op_type() == "Conv2D" || node.get_op_type() == "Conv3D") {
        if (tf_padding == "SAME") {
            // According to the formulas for calculating auto_pad values of the
            // Conv layer in the Operation specification,
            // the SAME_UPPER value matches to the SAME value in TensorFlow
            return ov::op::PadType::SAME_UPPER;
        }
    }

    return ov::op::PadType::EXPLICIT;
}
