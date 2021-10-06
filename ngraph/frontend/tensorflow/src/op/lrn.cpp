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
static Status TranslateLRNOp(const TFNodeDecoder* op,
                             const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>& static_input_map,
                             Builder::OpMap& ng_op_map) {
    Output<Node> ng_inp;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_inp));

    float alpha;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "alpha", &alpha));
    float beta;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "beta", &beta));
    float bias;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "bias", &bias));
    int64_t depth_radius;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "depth_radius", &depth_radius));

    // OV: Each input value is divided by (bias+(alpha/size)*sum(xi^2 for every xi
    // in the local region))^beta
    // TF: sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d +
    // depth_radius + 1] ** 2)
    //     output = input / (bias + alpha * sqr_sum) ** beta
    int64_t size = depth_radius * 2 + 1;
    alpha = alpha * size;
    // nGraph expects the input to be in NCHW format
    NHWCtoNCHW(node.get_name(), true, ng_inp);
    auto ng_output = ConstructNgNode<LRN>(node.get_name(), ng_inp, alpha, beta,
                                                 bias, (size_t)size);
    NCHWtoNHWC(node.get_name(), true, ng_output);
    SaveNgOp(ng_op_map, node.get_name(), ng_output);
    return Status::OK();

}
}
}
#endif