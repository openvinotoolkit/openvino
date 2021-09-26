// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

// Translate TopKV2 Op using ngraph core op TopK
namespace tensorflow {
namespace ngraph_bridge {

#if 0
OutputVector TranslateTopKV2Op(
        const NodeContext& node) {
    Output<ngraph::Node> ng_input;

    TF_RETURN_IF_ERROR(ValidateInputCount(op, 2));
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

    // axis along which to compute top k indices
    int64_t k_axis = ng_input.get_shape().size() - 1;

    // scalar input tensor specifying how many max/min elts should be computed
    // CPU backend only supports element type i64
    std::vector<int64_t> ng_k_vec;
    TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &ng_k_vec));
    auto ng_k = ConstructNgNode<opset::Constant>(node.get_name(), element::i64,
                                                 Shape{}, ng_k_vec[0]);

    std::string mode = "max";

    std::string sort = "value";
    bool sorted = true;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "sorted", &sorted));
    if (!sorted) {
        sort = "index";
    }

    auto ng_result =
            std::make_shared<opset::TopK>(ng_input, ng_k, k_axis, mode, sort);

    Output<Node> ng_values = ng_result->output(0);
    Builder::SetTracingInfo(node.get_name(), ng_values);
    Output<Node> ng_indices = ng_result->output(1);
    Builder::SetTracingInfo(node.get_name(), ng_indices);

    SaveNgOp(ng_op_map, node.get_name(), ng_values);
    SaveNgOp(ng_op_map, node.get_name(), ng_indices);

    return Status::OK();
}

#endif
}  // namespace ngraph_bridge
}  // namespace tensorflow