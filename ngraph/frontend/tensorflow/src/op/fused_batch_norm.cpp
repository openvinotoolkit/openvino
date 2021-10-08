// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateFusedBatchNormOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    auto ng_scale = node.get_ng_input(1);
    auto ng_offset = node.get_ng_input(2);
    auto ng_mean = node.get_ng_input(3);
    auto ng_variance = node.get_ng_input(4);

    bool is_v3 = node.get_op_type() == "FusedBatchNormV3";
    bool is_Ex = node.get_op_type() == "_FusedBatchNormEx";

    auto data_format = node.get_attribute<std::string>("data_format");
    TF_OP_VALIDATION_CHECK(node, data_format == "NHWC" || data_format == "NCHW", "Unsupported data format");

    bool is_nhwc = (data_format == "NHWC");

    NGRAPH_VLOG(3) << "data_format: " << data_format;

    // TODO: where does 0.0001 come from?
    auto tf_epsilon = node.get_attribute<float>("epsilon", 0.0001);

    NGRAPH_VLOG(3) << "epsilon: " << tf_epsilon;

    NHWCtoNCHW(node.get_name(), is_nhwc, ng_input);

    auto ng_batch_norm = ConstructNgNode<BatchNormInference>(node.get_name(),
                                                             ng_input,
                                                             ng_scale,
                                                             ng_offset,
                                                             ng_mean,
                                                             ng_variance,
                                                             tf_epsilon);
    NCHWtoNHWC(node.get_name(), is_nhwc, ng_batch_norm);

    if (is_Ex) {
        string activation_mode = node.get_attribute<string>("activation_mode");
        TF_OP_VALIDATION_CHECK(node, activation_mode == "Relu", "Unsupported _FusedBatchNormEx activation mode");
        auto relu_op = ConstructNgNode<Relu>(node.get_name(), ng_batch_norm);
        return {relu_op};
    } else {
        // TODO: Why are there so many? Is it correct?
        OutputVector result = {ng_batch_norm, ng_mean, ng_variance, ng_mean, ng_variance};
        if (is_v3) {
            // FusedBatchNormV3 has 6 outputs
            result.push_back(ng_mean);  // reserve_space_3
        }
        return result;
    }
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph