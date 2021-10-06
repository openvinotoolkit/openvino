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

static Status TranslateZerosLikeOp(const TFNodeDecoder* op,
                                   const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>&,
                                   Builder::OpMap& ng_op_map) {
  Output<Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input));

  Shape input_shape = ng_input.get_shape();
  std::vector<std::string> const_values(shape_size(input_shape), "0");
  auto ng_result = ConstructNgNode<Constant>(
      node.get_name(), ng_input.get_element_type(), input_shape, const_values);
  SaveNgOp(ng_op_map, node.get_name(), ng_result);
  return Status::OK();
}

}
}
#endif