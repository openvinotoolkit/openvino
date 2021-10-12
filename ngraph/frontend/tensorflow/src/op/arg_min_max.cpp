// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateArgMinMax(const NodeContext& node, std::string mode) {
    Output<Node> ng_input = node.get_ng_input(0);

    std::vector<int64_t> tf_dim;
    GetStaticInputVector(node, 1, &tf_dim);

    Shape input_shape = ng_input.get_shape();
    size_t input_rank = input_shape.size();

    if (tf_dim.size() != 1) {
        throw errors::InvalidArgument("ArgMax Op: dimension must be scalar, operates on a single axis");
    }

    // If input dimension is negative, make it positive
    if (tf_dim[0] < 0) {
        NGRAPH_VLOG(3) << "Input dimension is negative, make it positive " << tf_dim[0];
        tf_dim[0] = (int64_t)input_rank + tf_dim[0];
    }
    NGRAPH_VLOG(3) << "Axis along which to compute " << tf_dim[0];
    size_t k_axis = tf_dim[0];

    auto ng_et = node.get_attribute<element::Type>("output_type");

    auto ng_k = ConstructNgNode<Constant>(node.get_name(), element::i64, Shape{}, std::vector<int64_t>({1}));

    std::string sort = "none";
    auto ng_topk = std::make_shared<TopK>(ng_input, ng_k, k_axis, mode, sort, ng_et);
    auto ng_indices = ng_topk->output(1);
    int axis = ng_topk->get_axis();
    auto axis_to_remove =
        ConstructNgNode<Constant>(node.get_name(), element::i64, Shape{1}, std::vector<int64_t>({axis}));
    auto reshaped_indices = ConstructNgNode<Squeeze>(node.get_name(), ng_indices, axis_to_remove);
    SetTracingInfo(node.get_name(), reshaped_indices);
    return {reshaped_indices};
}

OutputVector TranslateArgMaxOp(const NodeContext& node) {
    return (TranslateArgMinMax(node, "max"));
}

OutputVector TranslateArgMinOp(const NodeContext& node) {
    return (TranslateArgMinMax(node, "min"));
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
