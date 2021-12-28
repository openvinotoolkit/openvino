// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion_extensions.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;
namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_fused_batch_norm_op(const ov::frontend::NodeContext& node) {
    auto ng_input = node.get_input(0);
    auto ng_scale = node.get_input(1);
    auto ng_offset = node.get_input(2);
    auto ng_mean = node.get_input(3);
    auto ng_variance = node.get_input(4);

    auto data_format = node.get_attribute<std::string>("data_format");
    FRONT_END_GENERAL_CHECK(data_format == "NHWC" || data_format == "NCHW", "Unsupported data format");
    bool is_nhwc = (data_format == "NHWC");
    NGRAPH_DEBUG << "data_format: " << data_format;
    // TODO: where does 0.0001 come from?
    auto tf_epsilon = node.get_attribute<float>("epsilon", 0.0001);
    NGRAPH_DEBUG << "epsilon: " << tf_epsilon;
    convert_nhwc_to_nchw(is_nhwc, ng_input);
    auto ng_batch_norm =
        make_shared<BatchNormInference>(ng_input, ng_scale, ng_offset, ng_mean, ng_variance, tf_epsilon)->output(0);
    convert_nchw_to_nhwc(is_nhwc, ng_batch_norm);

    string activation_mode = node.get_attribute<string>("activation_mode");
    FRONT_END_GENERAL_CHECK(activation_mode == "Relu", "Unsupported _FusedBatchNormEx activation mode");
    auto relu_op = make_shared<Relu>(ng_batch_norm);
    return {relu_op};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov