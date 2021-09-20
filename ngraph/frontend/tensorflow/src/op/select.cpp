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
static Status TranslateSelectOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
  Output<Node> ng_input1, ng_input2, ng_input3;
  TF_RETURN_IF_ERROR(
      GetInputNodes(ng_op_map, op, ng_input1, ng_input2, ng_input3));
  auto ng_select = ConstructNgNode<opset::Select>(node.get_name(), ng_input1,
                                                  ng_input2, ng_input3);
  SaveNgOp(ng_op_map, node.get_name(), ng_select);
  return Status::OK();
}
}
}

#endif
