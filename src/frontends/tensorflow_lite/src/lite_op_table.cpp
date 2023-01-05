// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lite_op_table.hpp"

#include "decoder_map.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"CONV_2D", conv2d},
        {"DEPTHWISE_CONV_2D", depthwise_conv2d},
        {"CONCATENATION", concatenation},
        {"RESHAPE", reshape},
        {"LOGISTIC", ov::frontend::tensorflow::op::translate_unary_op<opset10::Sigmoid>},
        {"RELU", ov::frontend::tensorflow::op::translate_unary_op<opset10::Relu>},
        {"PAD", pad},
        {"ADD", ov::frontend::tensorflow::op::translate_binary_op<opset10::Add>},
        // AVERAGE_POOL_2D
        // PACK
        // SOFTMAX
    };
}
}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov