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

static Status TranslateZerosLikeOp(const TFNodeDecoder* op,
                                   const std::vector<const ngraph::frontend::tensorflow::detail::TensorWrapper*>&,
                                   Builder::OpMap& ng_op_map) {
  Output<Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  Shape input_shape = ng_input.get_shape();
  std::vector<std::string> const_values(shape_size(input_shape), "0");
  auto ng_result = ConstructNgNode<opset::Constant>(
      node.get_name(), ng_input.get_element_type(), input_shape, const_values);
  SaveNgOp(ng_op_map, node.get_name(), ng_result);
  return Status::OK();
}

}
}
#endif