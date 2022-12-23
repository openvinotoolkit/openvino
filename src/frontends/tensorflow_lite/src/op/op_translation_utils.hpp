// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"
#include "op_table.hpp"
#include "../decoder_map.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {


void set_output_names(const ov::frontend::tensorflow::NodeContext& node, OutputVector& outputs);
void del_output_names(OutputVector& outputs);

// convolutions
template <class T>
std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap> get_conv_decoder_map(
        const std::string& option_names, const std::string& new_type_name, const ov::frontend::tensorflow::NodeContext& node) {
    const auto& decoder = node.get_decoder();
    const auto* conv_opts = decoder->get_attribute(option_names).as<const T*>();
    const std::map<std::string, ov::Any> attrs {
            {"strides", std::vector<int64_t>{1, conv_opts->stride_h(), conv_opts->stride_w(), 1}},
            {"padding", std::string(EnumNamePadding(conv_opts->padding()))},
            {"dilations", std::vector<int64_t>{1, conv_opts->dilation_h_factor(), conv_opts->dilation_w_factor(), 1}},
            {"data_format", "NHWC"},
            {"activation", EnumNameActivationFunctionType(conv_opts->fused_activation_function())},
    };
    return std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(decoder, attrs, new_type_name, true);
}
void get_conv(ov::OutputVector& output, const ov::frontend::tensorflow::NodeContext& node, const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder, ov::OutputVector(*converter)(const ov::frontend::tensorflow::NodeContext&));
void get_bias(ov::OutputVector& output, const ov::frontend::tensorflow::NodeContext& node, const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder);
void get_activation(ov::OutputVector& output, const ov::frontend::tensorflow::NodeContext& node, const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder);

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov