// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {
static Status TranslateRankOp(const TFNodeDecoder* op, const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                              Builder::OpMap& ng_op_map) {
    Output<Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

    Shape input_shape = ng_input.get_shape();
    auto input_rank = static_cast<int>(input_shape.size());

    auto ng_rank = ConstructNgNode<opset::Constant>(
            node.get_name(), element::i32, Shape(),
            std::vector<int>({input_rank}));

    SaveNgOp(ng_op_map, node.get_name(), ng_rank);
    return Status::OK();
}
}
}

#endif