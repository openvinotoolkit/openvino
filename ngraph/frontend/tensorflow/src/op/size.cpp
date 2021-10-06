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
static Status TranslateSizeOp(const TFNodeDecoder* op, const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
    Output<Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "out_type", &dtype));

    // Size has an attribute to specify output, int32_t or int64_t
    element::Type type;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &type));

    auto ng_input_shape = ng_input.get_shape();
    int64_t result = 1;
    for (auto dim : ng_input_shape) {
        result *= dim;
    }

    // make a scalar with value equals to result
    auto ng_result = ConstructNgNode<Constant>(
            node.get_name(), type, Shape(0), std::vector<int64_t>({result}));

    SaveNgOp(ng_op_map, node.get_name(), ng_result);
    return Status::OK();
}
}
}
#endif