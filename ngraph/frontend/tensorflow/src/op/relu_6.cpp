// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {
static Status TranslateRelu6Op(const TFNodeDecoder* op,
                               const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                               Builder::OpMap& ng_op_map) {
    Output<Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));
    SaveNgOp(ng_op_map, node.get_name(),
             ConstructNgNode<opset::Clamp>(node.get_name(), ng_input, 0, 6));
    return Status::OK();
}
}
}

#endif