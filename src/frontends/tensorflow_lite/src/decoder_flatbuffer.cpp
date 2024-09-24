// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_flatbuffer.h"

#ifdef FLATBUFFERS_LOCALE_INDEPENDENT
#    undef FLATBUFFERS_LOCALE_INDEPENDENT
#endif
#define FLATBUFFERS_LOCALE_INDEPENDENT 0
#include "flatbuffers/flexbuffers.h"
#include "schema_generated.h"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

namespace {
TensorMetaInfo extract_tensor_meta_info(const TensorInfo& tensor_info) {
    TensorMetaInfo tensor_meta_info;
    const auto tensor = tensor_info.tensor;
    const uint8_t* tensor_data =
        (tensor_info.buffer && tensor_info.buffer->data() ? tensor_info.buffer->data()->data() : nullptr);

    tensor_meta_info.m_partial_shape =
        ov::frontend::tensorflow_lite::get_ov_shape(tensor->shape(), tensor->shape_signature());
    tensor_meta_info.m_element_type = ov::frontend::tensorflow_lite::get_ov_type(tensor->type());
    tensor_meta_info.m_quantization_info = ov::frontend::tensorflow_lite::get_quantization(tensor->quantization());
    tensor_meta_info.m_sparsity_info = ov::frontend::tensorflow_lite::get_sparsity(tensor->shape(),
                                                                                   tensor->sparsity(),
                                                                                   tensor_meta_info.m_element_type,
                                                                                   tensor_data);
    tensor_meta_info.m_tensor_data = tensor_data;
    tensor_meta_info.m_tensor_name = tensor->name()->str();

    return tensor_meta_info;
}
}  // namespace

size_t DecoderFlatBuffer::get_input_size() const {
    return m_input_info.size();
}

void DecoderFlatBuffer::get_input_node(size_t input_port_idx,
                                       std::string& producer_name,
                                       std::string& producer_output_port_name,
                                       size_t& producer_output_port_index) const {
    const auto inputs = m_node_def->inputs();
    FRONT_END_GENERAL_CHECK(inputs->size() > input_port_idx,
                            "Input port index is out of range for node ",
                            get_op_name(),
                            ". Requested input index: ",
                            input_port_idx,
                            ". Number of inputs: ",
                            inputs->size());
    auto input_tensor_idx = (*inputs)[static_cast<flatbuffers::uoffset_t>(input_port_idx)];
    auto tensor = m_input_info.at(input_port_idx).tensor;
    std::string name = (*tensor).name()->str();
    producer_name = name;
    producer_output_port_index = input_tensor_idx;
}

const std::string& DecoderFlatBuffer::get_op_type() const {
    return m_type;
}

const std::string& DecoderFlatBuffer::get_op_name() const {
    return m_name;
}

size_t DecoderFlatBuffer::get_output_size() const {
    return m_node_def->outputs()->size();
}

std::string DecoderFlatBuffer::get_input_tensor_name(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    return m_input_info.at(idx).tensor->name()->str();
}

ov::element::Type DecoderFlatBuffer::get_input_tensor_type(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    return get_ov_type(m_input_info.at(idx).tensor->type());
}

std::string DecoderFlatBuffer::get_output_tensor_name(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return m_output_info.at(idx).tensor->name()->str();
}

ov::element::Type DecoderFlatBuffer::get_output_tensor_type(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return get_ov_type(m_output_info.at(idx).tensor->type());
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_input_tensor(
    size_t idx,
    const ov::frontend::InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    return decode_tensor(m_input_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_output_tensor(
    size_t idx,
    const ov::frontend::InputModel& model) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    return decode_tensor(m_output_info.at(idx), model);
}

std::shared_ptr<ov::frontend::tensorflow_lite::TensorLitePlace> DecoderFlatBuffer::decode_tensor(
    const ov::frontend::tensorflow_lite::TensorInfo& tensor_info,
    const ov::frontend::InputModel& model) const {
    const auto tensor = tensor_info.tensor;
    std::vector<std::string> names = {tensor->name()->str()};
    const uint8_t* tensor_data =
        (tensor_info.buffer && tensor_info.buffer->data() ? tensor_info.buffer->data()->data() : nullptr);

    return std::make_shared<ov::frontend::tensorflow_lite::TensorLitePlace>(
        model,
        ov::frontend::tensorflow_lite::get_ov_shape(tensor->shape(), tensor->shape_signature()),
        ov::frontend::tensorflow_lite::get_ov_type(tensor->type()),
        names,
        ov::frontend::tensorflow_lite::get_quantization(tensor->quantization()),
        ov::frontend::tensorflow_lite::get_sparsity(tensor->shape(),
                                                    tensor->sparsity(),
                                                    ov::frontend::tensorflow_lite::get_ov_type(tensor->type()),
                                                    tensor_data),
        tensor_data);
}

ov::Any get_value_as_ov_any(const flexbuffers::Reference& value) {
#define CASE_MACRO(fbt, as_stmt) \
    case flexbuffers::fbt:       \
        return {value.as_stmt()};
    switch (value.GetType()) {
        CASE_MACRO(FBT_INT, AsInt32)
        CASE_MACRO(FBT_INDIRECT_INT, AsInt32)
        CASE_MACRO(FBT_UINT, AsUInt32)
        CASE_MACRO(FBT_INDIRECT_UINT, AsUInt32)
        CASE_MACRO(FBT_FLOAT, AsFloat)
        CASE_MACRO(FBT_INDIRECT_FLOAT, AsFloat)
        CASE_MACRO(FBT_STRING, AsString)
        CASE_MACRO(FBT_BOOL, AsBool)
    default:
        return {};
    }
    return {};
}

ov::Any DecoderFlatBuffer::get_attribute(const std::string& name) const {
    if (name == "new_shape" && m_type == "RESHAPE") {
        bool has_attribute = this->has_attribute(&tflite::ReshapeOptions::new_shape);
        if (has_attribute) {
            auto reshape_new_shape = this->get_attribute(&tflite::ReshapeOptions::new_shape);
            const auto new_shape = std::vector<int64_t>(reshape_new_shape->begin(), reshape_new_shape->end());
            return new_shape;
        } else {
            return {};
        }
    } else if (name == "output_type" && m_type == "ARG_MIN") {
        bool has_attribute = this->has_attribute(&tflite::ArgMinOptions::output_type);
        if (has_attribute) {
            return get_ov_type(this->get_attribute(&tflite::ArgMinOptions::output_type));
        } else {
            return {};
        }
    } else if (name == "output_type" && m_type == "ARG_MAX") {
        bool has_attribute = this->has_attribute(&tflite::ArgMaxOptions::output_type);
        if (has_attribute) {
            return get_ov_type(this->get_attribute(&tflite::ArgMaxOptions::output_type));
        } else {
            return {};
        }
    } else if (name == "adj_x" && m_type == "BATCH_MATMUL") {
        bool has_attribute = this->has_attribute(&tflite::BatchMatMulOptions::adj_x);
        if (has_attribute) {
            return this->get_attribute(&tflite::BatchMatMulOptions::adj_x);
        } else {
            return {};
        }
    } else if (name == "adj_y" && m_type == "BATCH_MATMUL") {
        bool has_attribute = this->has_attribute(&tflite::BatchMatMulOptions::adj_y);
        if (has_attribute) {
            return this->get_attribute(&tflite::BatchMatMulOptions::adj_y);
        } else {
            return {};
        }
    } else if (name == "DstT" && m_type == "CAST") {
        return this->get_output_tensor_type(0);
    } else if (name == "axis" && m_type == "CONCATENATION") {
        return static_cast<int64_t>(this->get_attribute(&tflite::ConcatenationOptions::axis));
    } else if (name == "block_size" && m_type == "DEPTH_TO_SPACE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::DepthToSpaceOptions::block_size));
    } else if (name == "axis" && m_type == "GATHER") {
        return this->get_attribute(&tflite::GatherOptions::axis);
    } else if (name == "batch_dims" && m_type == "GATHER") {
        return this->get_attribute(&tflite::GatherOptions::batch_dims);
    } else if (name == "alpha" && m_type == "LEAKY_RELU") {
        return this->get_attribute(&tflite::LeakyReluOptions::alpha);
    } else if (name == "mode" && m_type == "MIRROR_PAD") {
        return std::string(EnumNameMirrorPadMode(this->get_attribute(&tflite::MirrorPadOptions::mode)));
    } else if (name == "axis" && m_type == "ONE_HOT") {
        return static_cast<int64_t>(this->get_attribute(&tflite::OneHotOptions::axis));
    } else if (name == "axis" && m_type == "PACK") {
        return static_cast<int64_t>(this->get_attribute(&tflite::PackOptions::axis));
    } else if (name == "align_corners" && m_type == "RESIZE_BILINEAR") {
        return this->get_attribute(&tflite::ResizeBilinearOptions::align_corners);
    } else if (name == "half_pixel_centers" && m_type == "RESIZE_BILINEAR") {
        return this->get_attribute(&tflite::ResizeBilinearOptions::half_pixel_centers);
    } else if (name == "align_corners" && m_type == "RESIZE_NEAREST_NEIGHBOR") {
        return this->get_attribute(&tflite::ResizeNearestNeighborOptions::align_corners);
    } else if (name == "half_pixel_centers" && m_type == "RESIZE_NEAREST_NEIGHBOR") {
        return false;
    } else if (name == "seq_dim" && m_type == "REVERSE_SEQUENCE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::ReverseSequenceOptions::seq_dim));
    } else if (name == "batch_dim" && m_type == "REVERSE_SEQUENCE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::ReverseSequenceOptions::batch_dim));
    } else if (name == "out_type" && m_type == "SHAPE") {
        return get_ov_type(this->get_attribute(&tflite::ShapeOptions::out_type));
    } else if (name == "beta" && m_type == "SOFTMAX") {
        return this->get_attribute(&tflite::SoftmaxOptions::beta);
    } else if (name == "block_size" && m_type == "SPACE_TO_DEPTH") {
        return static_cast<int64_t>(this->get_attribute(&tflite::SpaceToDepthOptions::block_size));
    } else if (name == "num_split" && m_type == "SPLIT") {
        return static_cast<int64_t>(this->get_attribute(&tflite::SplitOptions::num_splits));
    } else if (name == "axis" && m_type == "SQUEEZE") {
        auto squeeze_dims = this->get_attribute(&tflite::SqueezeOptions::squeeze_dims);
        std::vector<int64_t> axes{squeeze_dims->begin(), squeeze_dims->end()};
        return axes;
    } else if (name == "begin_mask" && m_type == "STRIDED_SLICE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::StridedSliceOptions::begin_mask));
    } else if (name == "end_mask" && m_type == "STRIDED_SLICE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::StridedSliceOptions::end_mask));
    } else if (name == "new_axis_mask" && m_type == "STRIDED_SLICE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::StridedSliceOptions::new_axis_mask));
    } else if (name == "ellipsis_mask" && m_type == "STRIDED_SLICE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::StridedSliceOptions::ellipsis_mask));
    } else if (name == "shrink_axis_mask" && m_type == "STRIDED_SLICE") {
        return static_cast<int64_t>(this->get_attribute(&tflite::StridedSliceOptions::shrink_axis_mask));
    } else if (name == "axis" && m_type == "UNPACK") {
        return static_cast<int64_t>(this->get_attribute(&tflite::UnpackOptions::axis));
    } else if (name == "num" && m_type == "UNPACK") {
        return static_cast<int64_t>(this->get_attribute(&tflite::UnpackOptions::num));
    } else if (name == "out_idx" && m_type == "UNIQUE") {
        return get_ov_type(this->get_attribute(&tflite::UniqueOptions::idx_out_type));
    } else if (name == "cond_subgraph_index" && m_type == "WHILE") {
        return this->get_attribute(&tflite::WhileOptions::cond_subgraph_index);
    } else if (name == "body_subgraph_index" && m_type == "WHILE") {
        return this->get_attribute(&tflite::WhileOptions::body_subgraph_index);
    } else if (name == "strides" && m_type == "CONV_2D") {
        return std::vector<int64_t>{1,
                                    this->get_attribute(&tflite::Conv2DOptions::stride_h),
                                    this->get_attribute(&tflite::Conv2DOptions::stride_w),
                                    1};
    } else if (name == "padding" && m_type == "CONV_2D") {
        return std::string(EnumNamePadding(this->get_attribute(&tflite::Conv2DOptions::padding)));
    } else if (name == "dilations" && m_type == "CONV_2D") {
        return std::vector<int64_t>{1,
                                    this->get_attribute(&tflite::Conv2DOptions::dilation_h_factor),
                                    this->get_attribute(&tflite::Conv2DOptions::dilation_w_factor),
                                    1};
    } else if (name == "data_format" && m_type == "CONV_2D") {
        return "NHWC";
    } else if (name == "activation" && m_type == "CONV_2D") {
        return EnumNameActivationFunctionType(this->get_attribute(&tflite::Conv2DOptions::fused_activation_function));
    } else if (name == "strides" && m_type == "DEPTHWISE_CONV_2D") {
        return std::vector<int64_t>{1,
                                    this->get_attribute(&tflite::DepthwiseConv2DOptions::stride_h),
                                    this->get_attribute(&tflite::DepthwiseConv2DOptions::stride_w),
                                    1};
    } else if (name == "padding" && m_type == "DEPTHWISE_CONV_2D") {
        return std::string(EnumNamePadding(this->get_attribute(&tflite::DepthwiseConv2DOptions::padding)));
    } else if (name == "dilations" && m_type == "DEPTHWISE_CONV_2D") {
        return std::vector<int64_t>{1,
                                    this->get_attribute(&tflite::DepthwiseConv2DOptions::dilation_h_factor),
                                    this->get_attribute(&tflite::DepthwiseConv2DOptions::dilation_w_factor),
                                    1};
    } else if (name == "data_format" && m_type == "DEPTHWISE_CONV_2D") {
        return "NHWC";
    } else if (name == "activation" && m_type == "DEPTHWISE_CONV_2D") {
        return EnumNameActivationFunctionType(
            this->get_attribute(&tflite::DepthwiseConv2DOptions::fused_activation_function));
    } else if (name == "group" && m_type == "DEPTHWISE_CONV_2D") {
        return this->get_attribute(&tflite::DepthwiseConv2DOptions::depth_multiplier);
    } else if (name == "strides" && (m_type == "MAX_POOL_2D" || m_type == "AVERAGE_POOL_2D")) {
        return std::vector<int64_t>{1,
                                    this->get_attribute(&tflite::Pool2DOptions::stride_h),
                                    this->get_attribute(&tflite::Pool2DOptions::stride_w),
                                    1};
    } else if (name == "padding" && (m_type == "MAX_POOL_2D" || m_type == "AVERAGE_POOL_2D")) {
        return std::string(EnumNamePadding(this->get_attribute(&tflite::Pool2DOptions::padding)));
    } else if (name == "ksize" && (m_type == "MAX_POOL_2D" || m_type == "AVERAGE_POOL_2D")) {
        return std::vector<int64_t>{1,
                                    this->get_attribute(&tflite::Pool2DOptions::filter_height),
                                    this->get_attribute(&tflite::Pool2DOptions::filter_width),
                                    1};
    } else if (name == "data_format" && (m_type == "MAX_POOL_2D" || m_type == "AVERAGE_POOL_2D")) {
        return "NHWC";
    } else if (name == "activation" && (m_type == "MAX_POOL_2D" || m_type == "AVERAGE_POOL_2D")) {
        return EnumNameActivationFunctionType(this->get_attribute(&tflite::Pool2DOptions::fused_activation_function));
    } else if (name == "weights_format" && m_type == "FULLY_CONNECTED") {
        return static_cast<int8_t>(this->get_attribute(&tflite::FullyConnectedOptions::weights_format));
    } else if (name == "keep_num_dims" && m_type == "FULLY_CONNECTED") {
        return this->get_attribute(&tflite::FullyConnectedOptions::keep_num_dims);
    } else if (name == "fused_activation_function" && m_type == "FULLY_CONNECTED") {
        return std::string(EnumNameActivationFunctionType(
            this->get_attribute(&tflite::FullyConnectedOptions::fused_activation_function)));
    } else if (name == "fused_activation_function" && m_type == "ADD") {
        return std::string(
            EnumNameActivationFunctionType(this->get_attribute(&tflite::AddOptions::fused_activation_function)));
    } else if (name == "fused_activation_function" && m_type == "DIV") {
        return std::string(
            EnumNameActivationFunctionType(this->get_attribute(&tflite::DivOptions::fused_activation_function)));
    } else if (name == "fused_activation_function" && m_type == "MUL") {
        return std::string(
            EnumNameActivationFunctionType(this->get_attribute(&tflite::MulOptions::fused_activation_function)));
    } else if (name == "fused_activation_function" && m_type == "SUB") {
        return std::string(
            EnumNameActivationFunctionType(this->get_attribute(&tflite::SubOptions::fused_activation_function)));
    } else if (name == "strides" && m_type == "TRANSPOSE_CONV") {
        return std::vector<int64_t>{1,
                                    this->get_attribute(&tflite::TransposeConvOptions::stride_h),
                                    this->get_attribute(&tflite::TransposeConvOptions::stride_w),
                                    1};
    } else if (name == "padding" && m_type == "TRANSPOSE_CONV") {
        return std::string(EnumNamePadding(this->get_attribute(&tflite::TransposeConvOptions::padding)));
    } else if (name == "data_format" && m_type == "TRANSPOSE_CONV") {
        return "NHWC";
    } else if (name == "dilations" && m_type == "TRANSPOSE_CONV") {
        return std::vector<int64_t>{1, 1, 1, 1};
    } else if (name == "keep_dims" &&
               (m_type == "MEAN" || m_type == "REDUCE_ALL" || m_type == "REDUCE_ANY" || m_type == "REDUCE_MAX" ||
                m_type == "REDUCE_MIN" || m_type == "REDUCE_PROD" || m_type == "SUM")) {
        return this->get_attribute(&tflite::ReducerOptions::keep_dims);
    } else if (name == "approximate" && m_type == "GELU") {
        bool has_attribute = this->has_attribute(&tflite::GeluOptions::approximate);
        if (has_attribute) {
            return this->get_attribute(&tflite::GeluOptions::approximate);
        } else {
            return {};
        }
    } else if (name == "exclusive" && m_type == "CUMSUM") {
        bool has_attribute = this->has_attribute(&tflite::CumsumOptions::exclusive);
        if (has_attribute) {
            return this->get_attribute(&tflite::CumsumOptions::exclusive);
        } else {
            return {};
        }
    } else if (name == "reverse" && m_type == "CUMSUM") {
        bool has_attribute = this->has_attribute(&tflite::CumsumOptions::reverse);
        if (has_attribute) {
            return this->get_attribute(&tflite::CumsumOptions::reverse);
        } else {
            return {};
        }
    }

    const auto opts = m_node_def->custom_options();
    if (opts == nullptr)
        return {};
    const flexbuffers::Map& m = flexbuffers::GetRoot(opts->Data(), opts->size()).AsMap();
    return get_value_as_ov_any(m[name]);
}

TensorMetaInfo DecoderFlatBuffer::get_input_tensor_info(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_input_size(), "Requested input is out-of-range");
    const auto& tensor_info = m_input_info.at(idx);
    return extract_tensor_meta_info(tensor_info);
}

TensorMetaInfo DecoderFlatBuffer::get_output_tensor_info(size_t idx) const {
    FRONT_END_GENERAL_CHECK(idx < get_output_size(), "Requested output is out-of-range");
    const auto& tensor_info = m_output_info.at(idx);
    return extract_tensor_meta_info(tensor_info);
}

DecoderFlatBufferTensors::DecoderFlatBufferTensors(const TensorInfo& tensor_info, int64_t input_idx, int64_t output_idx)
    : m_input_idx(input_idx),
      m_output_idx(output_idx) {
    m_tensor_meta_info = extract_tensor_meta_info(tensor_info);
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
