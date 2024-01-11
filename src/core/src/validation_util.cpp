// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/validation_util.hpp"

#include <algorithm>
#include <numeric>

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "ngraph/evaluator.hpp"
#include "ngraph/op/negative.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/ops.hpp"
#include "sequnce_generator.hpp"
#include "validation_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

namespace ngraph {
using ov::Dimension;

//
// Infers the output shape of a windowed reduction operation, where the data may be dilated and/or
// padded, and the reduction window may be strided and/or dilated.
//
// TODO(amprocte): The messages here would be a bit friendlier if we didn't say "after
// padding/after dilation" for cases where there is actually no padding/dilation.
//
PartialShape infer_windowed_reduction_output_shape(const Node* node,
                                                   const PartialShape& data_shape,
                                                   const Strides& data_dilation,
                                                   const CoordinateDiff& data_padding_below,
                                                   const CoordinateDiff& data_padding_above,
                                                   const PartialShape& window_shape,
                                                   const Strides& window_strides,
                                                   const Strides& window_dilation,
                                                   bool is_window_all_in_padding_allowed,
                                                   bool ceil_mode) {
    PartialShape data_shape_merged{PartialShape::dynamic()};

    NODE_VALIDATION_CHECK(
        node,
        data_shape_merged.merge_rank(data_shape.rank()) && data_shape_merged.merge_rank(data_dilation.size()) &&
            data_shape_merged.merge_rank(data_padding_below.size()) &&
            data_shape_merged.merge_rank(data_padding_above.size()) &&
            data_shape_merged.merge_rank(window_shape.rank()) && data_shape_merged.merge_rank(window_strides.size()) &&
            data_shape_merged.merge_rank(window_dilation.size()),
        "Ranks for data shape (",
        data_shape,
        "), data dilation (",
        data_dilation,
        "), padding below (",
        data_padding_below,
        "), padding above (",
        data_padding_above,
        "), window shape (",
        window_shape,
        "), window strides (",
        window_strides,
        "), and window dilation (",
        window_dilation,
        ") do not match.");

    PartialShape output_shape = PartialShape::dynamic(data_shape_merged.rank());
    if (output_shape.rank().is_static()) {
        for (int64_t i = 0; i < output_shape.rank().get_length(); i++) {
            NODE_VALIDATION_CHECK(node,
                                  data_dilation[i] > 0,
                                  "Data dilation (",
                                  data_dilation,
                                  ") has zero dimension at axis ",
                                  i,
                                  ".");
            NODE_VALIDATION_CHECK(node,
                                  window_strides[i] > 0,
                                  "Window strides (",
                                  window_strides,
                                  ") has zero dimension at axis ",
                                  i,
                                  ".");
            NODE_VALIDATION_CHECK(node,
                                  window_dilation[i] > 0,
                                  "Window dilation (",
                                  window_dilation,
                                  ") has zero dimension at axis ",
                                  i,
                                  ".");

            bool data_dim_static = data_shape.rank().is_static() && data_shape[i].is_static();
            bool window_dim_static = window_shape.rank().is_static() && window_shape[i].is_static();

            ptrdiff_t data_padded_dilated_dim = -1;
            if (data_dim_static) {
                data_padded_dilated_dim = (static_cast<int64_t>(data_dilation[i]) * (data_shape[i].get_length() - 1)) +
                                          1 + data_padding_below[i] + data_padding_above[i];
                NODE_VALIDATION_CHECK(node,
                                      data_padded_dilated_dim > 0,
                                      "Data shape after padding and dilation has dimension less than 1 (dim: ",
                                      data_padded_dilated_dim,
                                      ") at axis ",
                                      i,
                                      ".");
            }

            ptrdiff_t window_dilated_dim = -1;
            if (window_dim_static) {
                window_dilated_dim = static_cast<int64_t>(window_dilation[i]) * (window_shape[i].get_length() - 1) + 1;

                NODE_VALIDATION_CHECK(node,
                                      window_dilated_dim > 0,
                                      "Window after dilation has dimension less than 1 (dim: ",
                                      window_dilated_dim,
                                      ") at axis ",
                                      i,
                                      ".");

                NODE_VALIDATION_CHECK(node,
                                      is_window_all_in_padding_allowed || (window_dilated_dim > data_padding_below[i] &&
                                                                           window_dilated_dim > data_padding_above[i]),
                                      "Window after dilation is sometimes entirely in the padding area for axis ",
                                      i,
                                      " (dilated window dimension: ",
                                      window_dilated_dim,
                                      ", padding below dimension: ",
                                      data_padding_below[i],
                                      ", padding above dimension: ",
                                      data_padding_above[i],
                                      ") and this is not ",
                                      "allowed.");
            }

            if (data_dim_static && window_dim_static) {
                NODE_VALIDATION_CHECK(node,
                                      window_dilated_dim <= data_padded_dilated_dim,
                                      "Window after dilation has dimension (dim: ",
                                      window_dilated_dim,
                                      ") larger than the data shape after padding (dim: ",
                                      data_padded_dilated_dim,
                                      ") at axis ",
                                      i,
                                      ".");

                if (ceil_mode) {
                    output_shape[i] =
                        ceil_div(static_cast<size_t>(data_padded_dilated_dim) - static_cast<size_t>(window_dilated_dim),
                                 window_strides[i]) +
                        1;
                } else {
                    output_shape[i] =
                        ((static_cast<size_t>(data_padded_dilated_dim) - static_cast<size_t>(window_dilated_dim)) /
                         window_strides[i]) +
                        1;
                }
            }
        }
    }

    return output_shape;
}

struct ChannelShapedInputSpec {
    element::Type m_element_type;
    PartialShape m_shape;
    std::string m_input_name;
};

static std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward_helper(
    const Node* node,
    element::Type input_element_type,
    const PartialShape& input_shape,
    const std::vector<ChannelShapedInputSpec>& channel_shaped_inputs) {
    // Built up a slash-separated string naming all the channel-shaped inputs, for use in error
    // messages.
    std::stringstream ss;
    bool first = true;
    for (const auto& inp : channel_shaped_inputs) {
        if (!first) {
            ss << "/";
        }
        ss << inp.m_input_name;
        first = false;
    }
    std::string channel_input_names = ss.str();

    // Infer output element type.
    element::Type et_result{input_element_type};

    for (const auto& inp : channel_shaped_inputs) {
        NODE_VALIDATION_CHECK(node,
                              element::Type::merge(et_result, et_result, inp.m_element_type),
                              "Input element types do not match.");
    }

    NODE_VALIDATION_CHECK(node,
                          et_result.is_dynamic() || et_result.is_real(),
                          "Input element types must be floating-point. Got: ",
                          et_result);

    // Extract channel dimension from input shape.
    Dimension channel_dim{Dimension::dynamic()};

    Rank input_rank = input_shape.rank();
    if (input_rank.is_static()) {
        NODE_VALIDATION_CHECK(node,
                              input_rank.get_length() >= 2,
                              "Input argument must have rank of at least 2 (input argument shape: ",
                              input_shape,
                              ").");

        channel_dim = input_shape[1];
    }

    // Infer gamma/beta/mu/sigma shape, which must be consistent with a vector of size
    // "channel_dim".
    PartialShape channel_shape{PartialShape::dynamic()};

    for (const auto& inp : channel_shaped_inputs) {
        NODE_VALIDATION_CHECK(node,
                              PartialShape::merge_into(channel_shape, inp.m_shape),
                              "Shapes for ",
                              channel_input_names,
                              " do not match.");
    }

    NODE_VALIDATION_CHECK(node,
                          channel_shape.merge_rank(1),
                          "Shape for ",
                          channel_input_names,
                          " (",
                          channel_shape,
                          ") does not have rank 1.");

    NODE_VALIDATION_CHECK(node,
                          Dimension::merge(channel_dim, channel_dim, channel_shape[0]),
                          "Input channel dimension (",
                          channel_dim,
                          ") does not match shape for ",
                          channel_input_names,
                          " (",
                          channel_shape,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          channel_dim.is_dynamic() || channel_dim.get_length() >= 1,
                          "Channel count must be at least 1.");

    // Batch result shape is same as the input shape, except we may possibly have inferred more
    // information from the channel count via gamma/beta/etc.
    PartialShape batch_result_shape{input_shape};

    if (batch_result_shape.rank().is_static()) {
        batch_result_shape[1] = channel_dim;
    }

    return std::make_tuple(et_result, batch_result_shape, PartialShape{channel_dim});
}

std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward(const Node* node,
                                                                               element::Type input_element_type,
                                                                               element::Type gamma_element_type,
                                                                               element::Type beta_element_type,
                                                                               element::Type mean_element_type,
                                                                               element::Type variance_element_type,
                                                                               const PartialShape& input_shape,
                                                                               const PartialShape& gamma_shape,
                                                                               const PartialShape& beta_shape,
                                                                               const PartialShape& mean_shape,
                                                                               const PartialShape& variance_shape) {
    return infer_batch_norm_forward_helper(node,
                                           input_element_type,
                                           input_shape,
                                           {{gamma_element_type, gamma_shape, "gamma"},
                                            {beta_element_type, beta_shape, "beta"},
                                            {mean_element_type, mean_shape, "mean"},
                                            {variance_element_type, variance_shape, "variance"}});
}

namespace {
/// \brief Scalar variant describes value of an Output, for use in max shape determination
///
/// For tensor values, we use the maximum value in the tensor
struct MaxValue {
    /// \brief No information known about the output
    MaxValue() = default;
    /// \brief uint64_t assoiated with the output
    MaxValue(uint64_t value) : m_value(value) {}
    MaxValue(const std::vector<uint64_t>& slices, int64_t slice_axis) : m_slices(slices), m_slice_axis(slice_axis) {
        m_value = *max_element(m_slices.begin(), m_slices.end());
    }
    uint64_t m_value{std::numeric_limits<uint64_t>::max()};
    std::vector<uint64_t> m_slices;
    int64_t m_slice_axis{-1};
};

std::vector<MaxValue> exec_constant(Node* node, std::vector<MaxValue>& inputs) {
    auto result = MaxValue();
    auto op = ov::as_type<ov::op::v0::Constant>(node);
    auto element_type = op->get_output_element_type(0);
    if (element_type.is_integral()) {
        uint64_t max_val = 0;
        if (element_type.is_signed()) {
            for (auto elt : op->cast_vector<int64_t>()) {
                if (max_val < static_cast<uint64_t>(elt)) {
                    max_val = elt;
                }
            }
        } else {
            for (auto elt : op->cast_vector<uint64_t>()) {
                if (max_val < elt) {
                    max_val = elt;
                }
            }
        }
        result = MaxValue(max_val);
    }
    return {result};
}

std::vector<MaxValue> exec_minimum(Node* node, std::vector<MaxValue>& inputs) {
    uint64_t min_value = std::numeric_limits<uint64_t>::max();
    switch (node->get_output_element_type(0)) {
    case element::Type_t::i8:
        min_value = std::numeric_limits<int8_t>::max();
        break;
    case element::Type_t::i16:
        min_value = std::numeric_limits<int16_t>::max();
        break;
    case element::Type_t::i32:
        min_value = std::numeric_limits<int32_t>::max();
        break;
    case element::Type_t::i64:
        min_value = std::numeric_limits<int64_t>::max();
        break;
    case element::Type_t::u8:
        min_value = std::numeric_limits<uint8_t>::max();
        break;
    case element::Type_t::u16:
        min_value = std::numeric_limits<uint16_t>::max();
        break;
    case element::Type_t::u32:
        min_value = std::numeric_limits<uint32_t>::max();
        break;
    case element::Type_t::u64:
        min_value = std::numeric_limits<uint64_t>::max();
        break;
    default:
        break;
    }
    min_value = std::min(min_value, inputs.at(0).m_value);
    min_value = std::min(min_value, inputs.at(1).m_value);
    return {MaxValue(min_value)};
}

std::vector<MaxValue> exec_concat(Node* node, std::vector<MaxValue>& inputs) {
    auto op = ov::as_type<ov::op::v0::Concat>(node);
    std::vector<uint64_t> slice_maxen;
    for (const auto& input : inputs) {
        slice_maxen.push_back(input.m_value);
    }
    auto axis = op->get_concatenation_axis();
    return {MaxValue(slice_maxen, axis)};
}

std::vector<MaxValue> exec_reduce_min(Node* node, std::vector<MaxValue>& inputs) {
    auto data = inputs.at(0);
    if (data.m_slice_axis >= 0 && data.m_slices.size() > 1) {
        if (auto indices_const = ov::as_type<op::v0::Constant>(node->get_input_node_ptr(1))) {
            if (indices_const->get_output_element_type(0).is_integral()) {
                const auto& indices_shape = indices_const->get_output_shape(0);
                if (indices_shape == Shape{1}) {
                    auto indices = indices_const->cast_vector<int64_t>();
                    auto axis = indices.at(0);
                    if (axis == data.m_slice_axis) {
                        return {MaxValue(*min_element(data.m_slices.begin(), data.m_slices.end()))};
                    }
                }
            }
        }
    }
    // Noting we can do
    return {MaxValue(data.m_value)};
}

std::vector<MaxValue> exec_shape_of(Node* node, std::vector<MaxValue>& inputs) {
    const auto& inputPS = node->get_input_partial_shape(0);
    std::vector<uint64_t> shapeDims;
    for (int64_t i = 0; i < inputPS.rank().get_length(); i++) {
        if (inputPS[i].is_static()) {
            shapeDims.push_back(inputPS[i].get_length());
        } else {
            shapeDims.push_back(std::numeric_limits<uint64_t>::max());
        }
    }

    return {MaxValue(shapeDims, 0)};
}

std::vector<MaxValue> exec_gather(Node* node, std::vector<MaxValue>& inputs) {
    auto gather = ov::as_type<ov::op::v1::Gather>(node);

    const auto& indices = ov::as_type_ptr<op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
    const auto& axis = ov::as_type_ptr<op::v0::Constant>(node->input_value(2).get_node_shared_ptr());

    if (!indices || !axis) {
        return {MaxValue()};
    }

    if (gather->get_axis() != 0) {
        return {MaxValue()};
    }

    const auto& indicesVec = indices->cast_vector<int64_t>();
    if (indicesVec.size() != 1 || indicesVec[0] >= static_cast<int64_t>(inputs[0].m_slices.size())) {
        return {MaxValue()};
    }

    return {MaxValue(inputs[0].m_slices[indicesVec[0]])};
}

std::vector<MaxValue> exec_nop(Node* node, std::vector<MaxValue>& inputs) {
    return {inputs.at(0)};
}
}  // namespace

std::shared_ptr<Node> operator-(const Output<Node>& arg0) {
    return std::make_shared<op::Negative>(arg0);
}
}  // namespace ngraph

void ov::infer_auto_padding(const Shape& image_shape,
                            const Shape& filter_shape,
                            const Strides& filter_strides,
                            const Strides& filter_dilations,
                            const op::PadType pad_type,
                            CoordinateDiff& padding_above,
                            CoordinateDiff& padding_below) {
    ov::util::infer_auto_padding(image_shape,
                                 filter_shape,
                                 filter_strides,
                                 filter_dilations,
                                 pad_type,
                                 padding_above,
                                 padding_below);
}

namespace {
const auto normalize_axis_to = [](const int64_t& tensor_rank) {
    return [&tensor_rank](int64_t& axis) {
        if (axis < 0) {
            axis += tensor_rank;
        }
    };
};

std::string normalize_axis_error_msg(const int64_t& axis, const int64_t& lower, const int64_t& upper) {
    return std::string(" Parameter axis ")
        .append(std::to_string(axis))
        .append(" out of the tensor rank range [")
        .append(std::to_string(lower))
        .append(", ")
        .append(std::to_string(upper))
        .append("].");
}
}  // namespace

int64_t ov::util::normalize(const int64_t& value, const int64_t& max) {
    return (value < 0) ? value + max : value;
};

void ov::normalize_axes(const Node* node, const int64_t& tensor_rank, std::vector<int64_t>& axes) {
    ov::util::normalize_axes(node, tensor_rank, axes);
}

std::vector<size_t> ov::normalize_axes(const std::string& node_description,
                                       const std::vector<int64_t>& axes,
                                       const Rank& tensor_rank) {
    return ov::util::normalize_axes(node_description, axes, tensor_rank);
}

int64_t ov::normalize_axis(const Node* node, std::int64_t axis, const Rank& tensor_rank) {
    return ov::util::normalize_axis(node, axis, tensor_rank);
}

int64_t ov::normalize_axis(const std::string& node_description, std::int64_t axis, const Rank& tensor_rank) {
    return ov::util::normalize_axis(node_description, axis, tensor_rank);
}

int64_t ov::normalize_axis(const Node* node,
                           std::int64_t axis,
                           std::uint64_t tensor_rank,
                           std::int64_t axis_range_min,
                           std::int64_t axis_range_max) {
    return ov::util::normalize_axis(node, axis, tensor_rank, axis_range_min, axis_range_max);
}

int64_t ov::normalize_axis(const std::string& node_description,
                           std::int64_t axis,
                           std::uint64_t tensor_rank,
                           std::int64_t axis_range_min,
                           std::int64_t axis_range_max) {
    return ov::util::normalize_axis(node_description, axis, tensor_rank, axis_range_min, axis_range_max);
}

bool ov::evaluate_as_partial_shape(const Output<Node>& output, PartialShape& pshape) {
    return ov::util::evaluate_as_partial_shape(output, pshape);
}

bool ov::default_label_evaluator(const Node* node, TensorLabelVector& output_labels) {
    return ov::util::default_label_evaluator(node, output_labels);
}

std::shared_ptr<ov::op::v0::Constant> ov::get_constant_from_source(const Output<Node>& source) {
    return ov::util::get_constant_from_source(source);
}

bool ov::has_no_labels(const ov::TensorLabel& labels) {
    return ov::util::has_no_labels(labels);
}

void ov::generate_transpose_default_order(std::vector<int64_t>& axes_order, const size_t length) {
    ov::util::generate_transpose_default_order(axes_order, length);
}

bool ov::is_valid_axes_order(const std::vector<int64_t>& axes_order, const size_t size) {
    return ov::util::is_valid_axes_order(axes_order, size);
}

bool ov::util::are_unique(const std::vector<int64_t>& data) {
    return std::unordered_set<int64_t>(data.begin(), data.cend()).size() == data.size();
}

// clip value to min, max
int64_t ov::util::clip(const int64_t& value, const int64_t& min, const int64_t& max) {
    return std::min(std::max(value, min), max);
};

std::shared_ptr<ov::op::v0::Constant> ov::util::constantfold_subgraph(const Output<Node>& subgraph_sink) {
    if (const auto& c = ov::as_type_ptr<op::v0::Constant>(subgraph_sink.get_node_shared_ptr()))
        return c;

    const auto node = subgraph_sink.get_node();
    const auto num_inputs = node->get_input_size();
    if (num_inputs == 0)
        return nullptr;

    if (subgraph_sink.get_tensor().has_and_set_bound()) {
        const auto& lower = subgraph_sink.get_tensor().get_lower_value();
        return std::make_shared<ov::op::v0::Constant>(lower);
    }

    if (ov::is_type<op::util::ShapeOfBase>(node) && node->get_input_partial_shape(0).is_dynamic()) {
        return nullptr;
    }

    OutputVector inputs;
    inputs.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        auto constant = constantfold_subgraph(node->input_value(i));
        if (constant == nullptr)
            return nullptr;
        inputs.push_back(constant);
    }

    OutputVector outputs(node->get_output_size());
    if (!node->constant_fold(outputs, inputs))
        return nullptr;
    return ov::as_type_ptr<op::v0::Constant>(outputs[subgraph_sink.get_index()].get_node_shared_ptr());
}

//
// Infers the output batch shape for convolution fprop.
//
ov::PartialShape ov::infer_convolution_forward(const Node* node,
                                               const PartialShape& data_batch_shape,
                                               const Strides& data_dilation,
                                               const CoordinateDiff& data_padding_below,
                                               const CoordinateDiff& data_padding_above,
                                               const PartialShape& filters_shape,
                                               const Strides& filter_strides,
                                               const Strides& filter_dilation) {
    return ov::util::infer_convolution_forward(node,
                                               data_batch_shape,
                                               data_dilation,
                                               data_padding_below,
                                               data_padding_above,
                                               filters_shape,
                                               filter_strides,
                                               filter_dilation);
}

namespace ov {
namespace util {
using ov::op::v0::Constant;

std::shared_ptr<Constant> get_constant_from_source(const Output<Node>& source) {
    if (const auto& c = ov::as_type_ptr<Constant>(source.get_node_shared_ptr())) {
        return c;
    } else if (has_and_set_equal_bounds(source)) {
        return std::make_shared<Constant>(source.get_tensor().get_upper_value());
    } else {
        return {};
    }
}

template <class T>
Tensor make_tensor_of_max_value(const element::Type_t et) {
    Tensor t{et, Shape{}};
    *t.data<T>() = std::numeric_limits<T>::max();
    return t;
}

Tensor make_tensor_of_max_value(const element::Type_t et) {
    switch (et) {
    case element::boolean:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::boolean>>(et);
    case element::bf16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::bf16>>(et);
    case element::f16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::f16>>(et);
    case element::f32:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::f32>>(et);
    case element::f64:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::f64>>(et);
    case element::i8:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i8>>(et);
    case element::i16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i16>>(et);
    case element::i32:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i32>>(et);
    case element::i64:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i64>>(et);
    case element::u1:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u1>>(et);
    case element::u8:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u8>>(et);
    case element::u16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u16>>(et);
    case element::u32:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u32>>(et);
    case element::u64:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u64>>(et);
    default:
        return {};
    }
}

template <class T>
Tensor make_tensor_of_min_value(const element::Type_t et) {
    Tensor t{et, Shape{}};
    *t.data<T>() = std::numeric_limits<T>::min();
    return t;
}

Tensor make_tensor_of_min_value(const element::Type_t et) {
    switch (et) {
    case element::boolean:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::boolean>>(et);
    case element::bf16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::bf16>>(et);
    case element::f16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::f16>>(et);
    case element::f32:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::f32>>(et);
    case element::f64:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::f64>>(et);
    case element::i8:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i8>>(et);
    case element::i16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i16>>(et);
    case element::i32:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i32>>(et);
    case element::i64:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i64>>(et);
    case element::u1:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u1>>(et);
    case element::u8:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u8>>(et);
    case element::u16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u16>>(et);
    case element::u32:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u32>>(et);
    case element::u64:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u64>>(et);
    default:
        return {};
    }
}

std::shared_ptr<op::v0::Constant> get_constant_max_of_type(element::Type_t t) {
    auto tensor = ov::util::make_tensor_of_max_value(t);
    return tensor ? std::make_shared<op::v0::Constant>(tensor) : nullptr;
}

std::shared_ptr<op::v0::Constant> get_constant_min_of_type(element::Type_t t) {
    auto tensor = ov::util::make_tensor_of_min_value(t);
    return tensor ? std::make_shared<op::v0::Constant>(tensor) : nullptr;
}

std::shared_ptr<op::v0::Constant> get_constant_lowest_of_type(element::Type_t t) {
#define OPENVINO_TYPE_TO_LOWEST_CONST(t)                                                                               \
    case t:                                                                                                            \
        return op::v0::Constant::create(t,                                                                             \
                                        {},                                                                            \
                                        {std::numeric_limits<typename element_type_traits<t>::value_type>::lowest()}); \
        break

    switch (t) {
        OPENVINO_TYPE_TO_LOWEST_CONST(element::boolean);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::bf16);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::f16);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::f32);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::f64);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::i8);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::i16);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::i32);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::i64);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::u1);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::u8);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::u16);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::u32);
        OPENVINO_TYPE_TO_LOWEST_CONST(element::u64);

    case element::undefined:
    case element::dynamic:
    default:
        return nullptr;
    }
}

std::vector<PartialShape> get_tensors_partial_shapes(const TensorVector& tensors) {
    std::vector<PartialShape> shapes;
    shapes.reserve(tensors.size());
    for (const auto& t : tensors) {
        shapes.emplace_back(t.get_shape());
    }
    return shapes;
}

std::vector<PartialShape> get_node_input_partial_shapes(const Node& node) {
    std::vector<PartialShape> shapes;
    shapes.reserve(node.get_input_size());
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        shapes.push_back(node.get_input_partial_shape(i));
    }
    return shapes;
}

bool is_rank_compatible_any_of(const Rank& r, std::initializer_list<Rank> others) {
    return std::any_of(others.begin(), others.end(), [&r](const Rank& other) {
        return r.compatible(other);
    });
}

PartialShape infer_convolution_forward(const Node* node,
                                       const PartialShape& data_batch_shape,
                                       const Strides& data_dilation,
                                       const CoordinateDiff& data_padding_below,
                                       const CoordinateDiff& data_padding_above,
                                       const PartialShape& filters_shape,
                                       const Strides& filter_strides,
                                       const Strides& filter_dilation) {
    Rank data_batch_filters_rank{Rank::dynamic()};

    NODE_VALIDATION_CHECK(node,
                          Rank::merge(data_batch_filters_rank, data_batch_shape.rank(), filters_shape.rank()),
                          "Data batch and filters rank do not match (data batch shape: ",
                          data_batch_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          data_batch_filters_rank.is_dynamic() || data_batch_filters_rank.get_length() >= 3,
                          "Data batch and filters must have rank of at least 3 (one batch axis, ",
                          "one input-channel axis, and at least one spatial dimension) ",
                          "(data batch shape: ",
                          data_batch_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    Rank spatial_rank{Rank::dynamic()};
    NODE_VALIDATION_CHECK(node,
                          Rank::merge(spatial_rank, spatial_rank, data_batch_filters_rank - 2) &&
                              Rank::merge(spatial_rank, spatial_rank, data_dilation.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, data_padding_below.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, data_padding_above.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, filter_strides.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, filter_dilation.size()),
                          "Ranks for data item shape/filters shape (data batch has shape ",
                          data_batch_shape,
                          ", so data item rank is ",
                          (data_batch_shape.rank() - 2),
                          " and filters have shape ",
                          filters_shape,
                          ", so filters spatial rank is ",
                          (filters_shape.rank() - 2),
                          "), data dilation (",
                          data_dilation,
                          "), padding below (",
                          data_padding_below,
                          "), padding above (",
                          data_padding_above,
                          "), filter strides (",
                          filter_strides,
                          "), and filter dilation (",
                          filter_dilation,
                          ") do not match.");

    Dimension batch_size = (data_batch_shape.rank().is_static() ? data_batch_shape[0] : Dimension::dynamic());
    Dimension data_channel_count = (data_batch_shape.rank().is_static() ? data_batch_shape[1] : Dimension::dynamic());
    PartialShape data_spatial_shape(PartialShape::dynamic(spatial_rank));

    Dimension filter_output_channel_count =
        (filters_shape.rank().is_static() ? filters_shape[0] : Dimension::dynamic());
    Dimension filter_input_channel_count = (filters_shape.rank().is_static() ? filters_shape[1] : Dimension::dynamic());
    PartialShape filter_spatial_shape(PartialShape::dynamic(spatial_rank));

    //
    // Note: spatial_rank is definitely static at this point.
    //

    for (int64_t i = 0; i < spatial_rank.get_length(); i++) {
        if (data_batch_shape.rank().is_static()) {
            data_spatial_shape[i] = data_batch_shape[i + 2];
        }

        if (filters_shape.rank().is_static()) {
            filter_spatial_shape[i] = filters_shape[i + 2];
        }
    }

    NODE_VALIDATION_CHECK(node, batch_size.is_dynamic() || batch_size.get_length() > 0, "Batch size is zero.");

    Dimension merged_channel_count;

    NODE_VALIDATION_CHECK(node,
                          Dimension::merge(merged_channel_count, data_channel_count, filter_input_channel_count),
                          "Data batch channel count (",
                          data_channel_count,
                          ") does not match filter input ",
                          "channel count (",
                          filter_input_channel_count,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          merged_channel_count.is_dynamic() || merged_channel_count.get_length() > 0,
                          "Data batch channel count and/or filter input channel count is zero.");

    NODE_VALIDATION_CHECK(node,
                          filter_output_channel_count.is_dynamic() || filter_output_channel_count.get_length() > 0,
                          "Filter output channel count is zero.");

    PartialShape data_output_shape = ngraph::infer_windowed_reduction_output_shape(node,
                                                                                   data_spatial_shape,
                                                                                   data_dilation,
                                                                                   data_padding_below,
                                                                                   data_padding_above,
                                                                                   filter_spatial_shape,
                                                                                   filter_strides,
                                                                                   filter_dilation,
                                                                                   true);

    PartialShape batch_output_shape(PartialShape::dynamic(spatial_rank + 2));
    batch_output_shape[0] = batch_size;
    batch_output_shape[1] = filter_output_channel_count;

    for (int64_t i = 0; i < spatial_rank.get_length(); i++) {
        batch_output_shape[i + 2] = data_output_shape[i];
    }

    return batch_output_shape;
}

bool try_apply_auto_padding(const PartialShape& image_shape,
                            const Shape& filter_shape,
                            const Strides& filter_strides,
                            const Strides& filter_dilations,
                            const op::PadType pad_type,
                            CoordinateDiff& padding_above,
                            CoordinateDiff& padding_below) {
    OPENVINO_ASSERT(pad_type == op::PadType::SAME_UPPER || pad_type == op::PadType::SAME_LOWER);

    if (image_shape.rank().is_dynamic()) {
        return false;
    }
    const auto image_dims = static_cast<std::vector<Dimension>>(image_shape);
    for (size_t i = 0; i < static_cast<size_t>(filter_shape.size()); i++) {
        if (image_dims[i + 2].is_static()) {
            auto image_size = static_cast<int64_t>(image_dims[i + 2].get_length());
            int64_t filter_size = (static_cast<int64_t>(filter_shape[i]) - 1) * filter_dilations[i] + 1;
            auto filter_stride = static_cast<int64_t>(filter_strides[i]);
            auto output_size = (image_size + filter_stride - 1) / filter_stride;

            auto padding_needed = std::max(int64_t(0), (output_size - 1) * filter_stride + filter_size - image_size);
            auto padding_lhs = padding_needed / 2;
            auto padding_rhs = padding_needed - padding_lhs;
            padding_below.push_back(pad_type == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs);
            padding_above.push_back(pad_type == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs);
        } else {
            padding_below.push_back(0);
            padding_above.push_back(0);
        }
    }
    return true;
}

void infer_auto_padding(const Shape& image_shape,
                        const Shape& filter_shape,
                        const Strides& filter_strides,
                        const Strides& filter_dilations,
                        const op::PadType pad_type,
                        CoordinateDiff& padding_above,
                        CoordinateDiff& padding_below) {
    const auto image_dims = std::vector<Dimension>(std::begin(image_shape), std::end(image_shape));
    // because image_shape is fully known result of try_apply_infer_auto_padding is ignored
    try_apply_auto_padding(image_dims,
                           filter_shape,
                           filter_strides,
                           filter_dilations,
                           pad_type,
                           padding_above,
                           padding_below);
}

bool evaluate_as_partial_shape(const Output<Node>& output, PartialShape& pshape) {
    Tensor lb, ub;
    std::tie(lb, ub) = evaluate_both_bounds(output);
    bool shape_defined = false;
    if (lb && ub) {
        auto lower_bound = std::make_shared<op::v0::Constant>(lb.get_element_type(), lb.get_shape(), lb.data())
                               ->cast_vector<int64_t>();
        auto upper_bound = std::make_shared<op::v0::Constant>(ub.get_element_type(), ub.get_shape(), ub.data())
                               ->cast_vector<int64_t>();
        OPENVINO_ASSERT(lower_bound.size() == upper_bound.size());
        const TensorLabel& labels = output.get_tensor().get_value_label();
        OPENVINO_ASSERT(labels.empty() || lower_bound.size() == labels.size());

        std::vector<Dimension> resulting_pshape(lower_bound.size());
        for (size_t i = 0; i < lower_bound.size(); ++i) {
            auto low = lower_bound[i], up = upper_bound[i];
            OPENVINO_ASSERT(low >= 0 && up >= 0, "Value for partial shape evaluation can't be lower than zero.");
            if (output.get_element_type() == element::i32 && low != up) {
                if (up == std::numeric_limits<std::int32_t>::max())
                    up = std::numeric_limits<std::int64_t>::max();
                if (low == std::numeric_limits<std::int32_t>::max())
                    low = std::numeric_limits<std::int64_t>::max();
            }
            resulting_pshape[i] = {low, up};
            if (!labels.empty() && labels[i])
                DimensionTracker::set_label(resulting_pshape[i], labels[i]);
        }
        pshape = PartialShape(resulting_pshape);
        shape_defined = true;
    }
    return shape_defined;
}

bool default_label_evaluator(const Node* node, TensorLabelVector& output_labels) {
    return default_label_evaluator(node, {0}, output_labels);
}

void generate_transpose_default_order(std::vector<int64_t>& axes_order, const size_t length) {
    axes_order.reserve(axes_order.size() + length);
    std::generate_n(std::back_inserter(axes_order), length, ov::SeqGen<size_t, ov::Direction::BACKWARD>(length - 1));
}

bool is_valid_axes_order(const std::vector<int64_t>& axes_order, const size_t size) {
    return are_unique(axes_order) &&
           std::all_of(axes_order.cbegin(), axes_order.cend(), ov::cmp::Between<int64_t, ov::cmp::LOWER>(0, size));
}

bool has_no_labels(const ov::TensorLabel& labels) {
    return std::all_of(labels.cbegin(), labels.cend(), cmp::Equal<size_t>(no_label));
}

std::vector<size_t> normalize_axes(const std::string& node_description,
                                   const std::vector<int64_t>& axes,
                                   const Rank& tensor_rank) {
    std::vector<size_t> new_axes;
    new_axes.reserve(axes.size());
    for (const auto& axis : axes) {
        new_axes.push_back(ov::util::normalize_axis(node_description, axis, tensor_rank));
    }
    return new_axes;
}

void normalize_axes(const Node* node, const int64_t& tensor_rank, std::vector<int64_t>& axes) {
    const auto axis_checker = cmp::Between<int64_t, cmp::BOTH>(-tensor_rank, tensor_rank ? (tensor_rank - 1) : 0);
    const auto invalid_axis = std::find_if_not(axes.cbegin(), axes.cend(), axis_checker);
    NODE_VALIDATION_CHECK(node,
                          invalid_axis == axes.cend(),
                          normalize_axis_error_msg(*invalid_axis, axis_checker.lower(), axis_checker.upper()));
    std::for_each(axes.begin(), axes.end(), normalize_axis_to(tensor_rank));
}

int64_t normalize_axis(const Node* node, std::int64_t axis, const Rank& tensor_rank) {
    return ov::util::normalize_axis(node->description(), axis, tensor_rank);
}

int64_t normalize_axis(const std::string& node_description, std::int64_t axis, const Rank& tensor_rank) {
    if (axis < 0) {
        // Handling negative axis requires static tensor rank
        OPENVINO_ASSERT(tensor_rank.is_static(),
                        node_description,
                        " Rank must be static in order to normalize negative axis=",
                        axis);
    }
    if (tensor_rank.is_dynamic()) {
        return axis;
    }

    const auto tensor_rank_value = tensor_rank.get_length();
    return normalize_axis(node_description,
                          axis,
                          tensor_rank_value,
                          -tensor_rank_value,
                          tensor_rank_value ? (tensor_rank_value - 1) : 0);
}

int64_t normalize_axis(const Node* node,
                       std::int64_t axis,
                       std::uint64_t tensor_rank,
                       std::int64_t axis_range_min,
                       std::int64_t axis_range_max) {
    return normalize_axis(node->description(), axis, tensor_rank, axis_range_min, axis_range_max);
}

int64_t normalize_axis(const std::string& node_description,
                       std::int64_t axis,
                       std::uint64_t tensor_rank,
                       std::int64_t axis_range_min,
                       std::int64_t axis_range_max) {
    // Accepted range of value for axis is [axis_range_min, axis_range_max].
    OPENVINO_ASSERT((axis_range_min <= axis) && (axis <= axis_range_max),
                    node_description,
                    normalize_axis_error_msg(axis, axis_range_min, axis_range_max));
    return normalize(axis, tensor_rank);
}
}  // namespace util
}  // namespace ov
