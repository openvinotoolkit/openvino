// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/util/log.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_fused_batch_norm_op(const NodeContext& node) {
    default_op_checks(node, 5, {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3"});
    auto ng_input = node.get_input(0);
    auto ng_scale = node.get_input(1);
    auto ng_offset = node.get_input(2);
    auto ng_mean = node.get_input(3);
    auto ng_variance = node.get_input(4);

    bool is_v3 = node.get_op_type() == "FusedBatchNormV3";

    auto data_format = node.get_attribute<std::string>("data_format");
    TENSORFLOW_OP_VALIDATION(node, data_format == "NHWC" || data_format == "NCHW", "Unsupported data format");

    bool is_nhwc = (data_format == "NHWC");

    OPENVINO_DEBUG << "data_format: " << data_format;

    // TODO: where does 0.0001 come from?
    auto tf_epsilon = node.get_attribute<float>("epsilon", 0.0001f);

    OPENVINO_DEBUG << "epsilon: " << tf_epsilon;

    convert_nhwc_to_nchw(is_nhwc, ng_input, ov::Rank(4));

    auto ng_batch_norm =
        make_shared<BatchNormInference>(ng_input, ng_scale, ng_offset, ng_mean, ng_variance, tf_epsilon)->output(0);
    convert_nchw_to_nhwc(is_nhwc, ng_batch_norm, ov::Rank(4));

    // TODO: Why are there so many? Is it correct?
    OutputVector result = {ng_batch_norm, ng_mean, ng_variance, ng_mean, ng_variance};
    if (is_v3) {
        // FusedBatchNormV3 has 6 outputs
        result.push_back(ng_mean);  // reserve_space_3
    }
    set_node_name(node.get_name(), ng_batch_norm.get_node_shared_ptr());
    return result;
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov