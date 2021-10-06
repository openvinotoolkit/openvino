// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

#if 0

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateReshapeOp(
        const NodeContext& node) {
    Output<Node> ng_input, ng_shape_op;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_shape_op));

    NGRAPH_VLOG(3) << "Input shape: " << join(ng_input.get_shape());

    std::vector<int64_t> shape;
    TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &shape));

    NGRAPH_VLOG(3) << "Requested result shape: " << join(shape);

    auto ng_shape = ConstructNgNode<Constant>(
            node.get_name(), element::i64, Shape{shape.size()}, shape);
    SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<Reshape>(
            node.get_name(), ng_input, ng_shape, false));
    return Status::OK();
}
}
}

#endif