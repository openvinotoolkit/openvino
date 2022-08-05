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
