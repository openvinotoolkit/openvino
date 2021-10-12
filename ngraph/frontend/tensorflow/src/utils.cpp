// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

void ngraph::frontend::tf::SetTracingInfo(const std::string& op_name, const ngraph::Output<ngraph::Node>& ng_node) {
    auto node = ng_node.get_node_shared_ptr();
    node->set_friendly_name(op_name);
    node->add_provenance_tag(op_name);
}

void ngraph::frontend::tf::TFTensorShapeToNGraphShape(const tensorflow::TensorShapeProto& tf_shape,
                                                      ngraph::PartialShape* ng_shape) {
    std::vector<ngraph::Dimension> dims;
    for (int i = 0; i < tf_shape.dim_size(); i++) {
        dims.push_back(tf_shape.dim(i).size());
    }
    *ng_shape = ngraph::PartialShape(dims);
}
