// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "../decoder_map.hpp"
#include "op_table.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

void set_output_names(const ov::frontend::tensorflow::NodeContext& node, OutputVector& outputs);
void del_output_names(OutputVector& outputs);

// convolutions
template <class T>
std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap> get_conv_decoder_map(
    const std::string& new_type_name,
    const ov::frontend::tensorflow::NodeContext& node) {
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");

    const std::map<std::string, ov::Any> attrs{
        {"strides",
         std::vector<int64_t>{1, decoder->get_attribute(&T::stride_h), decoder->get_attribute(&T::stride_w), 1}},
        {"padding", std::string(EnumNamePadding(decoder->get_attribute(&T::padding)))},
        {"dilations",
         std::vector<int64_t>{1,
                              decoder->get_attribute(&T::dilation_h_factor),
                              decoder->get_attribute(&T::dilation_w_factor),
                              1}},
        {"data_format", "NHWC"},
        {"activation", EnumNameActivationFunctionType(decoder->get_attribute(&T::fused_activation_function))},
    };
    return std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(node.get_decoder(), attrs, new_type_name, true);
}
void get_conv(ov::OutputVector& output,
              const ov::frontend::tensorflow::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::tensorflow::NodeContext&));
void get_bias(ov::OutputVector& output,
              const ov::frontend::tensorflow::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder);
void get_activation(ov::OutputVector& output,
                    const ov::frontend::tensorflow::NodeContext& node,
                    const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder);

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov