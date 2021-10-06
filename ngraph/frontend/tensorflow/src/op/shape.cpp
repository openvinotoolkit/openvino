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
static Status TranslateShapeOp(const TFNodeDecoder* op,
                               const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>&,
                               Builder::OpMap& ng_op_map) {
    Output<Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

    element::Type type;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

    // default output_type = element::i64
    SaveNgOp(ng_op_map, node.get_name(),
             ConstructNgNode<ShapeOf>(node.get_name(), ng_input, type));
    return Status::OK();
}
}
}

#endif