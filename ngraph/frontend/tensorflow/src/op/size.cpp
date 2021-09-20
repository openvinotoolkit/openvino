// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {
static Status TranslateSizeOp(const TFNodeDecoder* op, const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
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
    auto ng_result = ConstructNgNode<opset::Constant>(
            node.get_name(), type, Shape(0), std::vector<int64_t>({result}));

    SaveNgOp(ng_op_map, node.get_name(), ng_result);
    return Status::OK();
}
}
}
#endif