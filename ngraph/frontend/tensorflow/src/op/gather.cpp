// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

// See .../tensorflow/include/tensorflow/cc/ops/array_ops.h
// and .../openvino/ngraph/core/include/ngraph/op/gather.hpp
namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateGatherOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_input_indices = node.get_ng_input(1);

    auto ng_axis = ConstructNgNode<opset::Constant>(node.get_name(), element::i64, Shape{}, 0);

    auto gather_op = ConstructNgNode<opset::Gather>(node.get_name(), ng_input, ng_input_indices, ng_axis);

    return {gather_op};
}

OutputVector TranslateGatherV2Op(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_input_coords = node.get_ng_input(1);

    std::vector<int64_t> tf_axis;
    GetStaticInputVector(node, 2, &tf_axis);

    if (tf_axis.size() > 1) {
        std::ostringstream buf;
        buf << "Found axis in GatherV2 op (" << node.get_name() << ") translation to be non scalar, of size "
            << tf_axis.size();
        throw errors::Internal(buf.str());
    }

    // Negative axis is supported. Accounting for that
    auto ng_input_shape = ng_input.get_shape();
    size_t ng_input_rank = ng_input_shape.size();
    int axis;
    if (tf_axis[0] >= 0) {
        axis = tf_axis[0];
    } else {
        axis = tf_axis[0] + ng_input_rank;
    }
    if (axis < 0 || axis >= ng_input_rank) {
        std:
        ostringstream buf;
        buf << "Expected axis in the range [-" << ng_input_rank << ", " << ng_input_rank << "), but got " << tf_axis[0];
        throw errors::InvalidArgument(buf.str());
    }

    auto ng_axis =
            ConstructNgNode<opset::Constant>(node.get_name(), element::i64, Shape{tf_axis.size()}, tf_axis);

    auto gather_op = ConstructNgNode<opset::Gather>(node.get_name(), ng_input, ng_input_coords, ng_axis);

    return {gather_op};
}
}
}