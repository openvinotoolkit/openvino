// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

void ov::frontend::tf::SetTracingInfo(const std::string& op_name, const ov::Output<ov::Node> ng_node) {
    auto node = ng_node.get_node_shared_ptr();
    node->set_friendly_name(op_name);
    node->add_provenance_tag(op_name);
}

void ov::frontend::tf::TFTensorShapeToNGraphShape(const tensorflow::TensorShapeProto& tf_shape,
                                                  ov::PartialShape* ng_shape) {
    std::vector<ov::Dimension> dims;
    for (int i = 0; i < tf_shape.dim_size(); i++) {
        dims.emplace_back(tf_shape.dim(i).size());
    }
    *ng_shape = ov::PartialShape(dims);
}

void ov::frontend::tf::SetNodeNames(const std::string& node_name, const std::shared_ptr<Node>& node) {
    const auto& outputs = node->outputs();
    node->set_friendly_name(node_name);
    if (outputs.size() == 1) {
        SetOutputName(node_name, outputs[0]);
    }
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
        SetOutputName({node_name + ":" + std::to_string(idx)}, outputs[idx]);
    }
}

void ov::frontend::tf::SetOutputName(const std::string& out_name, const ov::Output<ov::Node>& output) {
    output.get_tensor().add_names({out_name});
}
