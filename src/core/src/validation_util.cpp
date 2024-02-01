// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/validation_util.hpp"

#include <algorithm>
#include <numeric>

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/ops.hpp"
#include "sequnce_generator.hpp"
#include "validation_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

namespace ngraph {
using ov::Dimension;
namespace op {
namespace v0 {
using ov::op::v0::Constant;
using ov::op::v0::Negative;
}  // namespace v0
}  // namespace op

Strides conv_default_strides(const Node* /* node */,
                             const ov::PartialShape& data_batch_shape,
                             const ov::PartialShape& filters_shape) {
    size_t rank;

    if (data_batch_shape.rank().is_static() && data_batch_shape.rank().get_length() >= 2) {
        rank = data_batch_shape.rank().get_length() - 2;
    } else if (filters_shape.rank().is_static() && filters_shape.rank().get_length() >= 2) {
        rank = filters_shape.rank().get_length() - 2;
    } else {
        rank = 0;
    }

    return Strides(rank, 1);
}

CoordinateDiff conv_default_padding(const Node* /* node */,
                                    const ov::PartialShape& data_batch_shape,
                                    const ov::PartialShape& filters_shape) {
    size_t rank;

    if (data_batch_shape.rank().is_static() && data_batch_shape.rank().get_length() >= 2) {
        rank = data_batch_shape.rank().get_length() - 2;
    } else if (filters_shape.rank().is_static() && filters_shape.rank().get_length() >= 2) {
        rank = filters_shape.rank().get_length() - 2;
    } else {
        rank = 0;
    }

    return CoordinateDiff(rank, 0);
}

//
// Infers the output shape of a windowed reduction operation, where the data may be dilated and/or
// padded, and the reduction window may be strided and/or dilated.
//
// TODO(amprocte): The messages here would be a bit friendlier if we didn't say "after
// padding/after dilation" for cases where there is actually no padding/dilation.
//
ov::PartialShape infer_windowed_reduction_output_shape(const Node* node,
                                                       const ov::PartialShape& data_shape,
                                                       const Strides& data_dilation,
                                                       const CoordinateDiff& data_padding_below,
                                                       const CoordinateDiff& data_padding_above,
                                                       const ov::PartialShape& window_shape,
                                                       const Strides& window_strides,
                                                       const Strides& window_dilation,
                                                       bool is_window_all_in_padding_allowed,
                                                       bool ceil_mode) {
    ov::PartialShape data_shape_merged{ov::PartialShape::dynamic()};

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

    ov::PartialShape output_shape = ov::PartialShape::dynamic(data_shape_merged.rank());
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

void validate_conv_params_spatial_dimensions(const Node* node,
                                             const size_t num_spatial_dims,
                                             const ov::op::PadType auto_pad,
                                             Strides& strides,
                                             Strides& dilations,
                                             CoordinateDiff& pads_begin,
                                             CoordinateDiff& pads_end) {
    if (strides.size() == 0) {
        strides = Strides(num_spatial_dims, 1);
    }
    if (dilations.size() == 0) {
        dilations = Strides(num_spatial_dims, 1);
    }
    if (pads_begin.size() == 0 || auto_pad == ov::op::PadType::VALID) {
        pads_begin = CoordinateDiff(num_spatial_dims, 0);
    }
    if (pads_end.size() == 0 || auto_pad == ov::op::PadType::VALID) {
        pads_end = CoordinateDiff(num_spatial_dims, 0);
    }
    NODE_VALIDATION_CHECK(node,
                          strides.size() == num_spatial_dims,
                          "Strides should be defined for all and only spatial features.");
    NODE_VALIDATION_CHECK(node,
                          dilations.size() == num_spatial_dims,
                          "Dilations should be defined for all and only spatial features.");
    NODE_VALIDATION_CHECK(node,
                          pads_begin.size() == num_spatial_dims && pads_end.size() == num_spatial_dims,
                          "Pads should be defined for all and only spatial features.");
}

//
// Infers the output batch shape and element type for batched pooling fprop.
//
ov::PartialShape infer_batched_pooling_forward(const Node* node,
                                               const ov::PartialShape& data_batch_shape,
                                               const CoordinateDiff& data_padding_below,
                                               const CoordinateDiff& data_padding_above,
                                               const ov::PartialShape& window_shape,
                                               const Strides& window_strides,
                                               bool is_window_all_in_padding_allowed,
                                               bool ceil_mode,
                                               const Strides& window_dilation) {
    NODE_VALIDATION_CHECK(node,
                          data_batch_shape.rank().is_dynamic() ||
                              (data_batch_shape.rank().get_length() >= 3 && data_batch_shape.rank().get_length() <= 5),
                          "Data batch must have rank of at least 4 or 5 (one batch axis, ",
                          "one input-channel axis, and two or three spatial dimension) ",
                          "(data batch shape: ",
                          data_batch_shape,
                          ").");

    ov::PartialShape data_spatial_shape{ov::PartialShape::dynamic()};

    NODE_VALIDATION_CHECK(node,
                          data_spatial_shape.merge_rank(data_batch_shape.rank() - 2) &&
                              data_spatial_shape.merge_rank(data_padding_below.size()) &&
                              data_spatial_shape.merge_rank(data_padding_above.size()) &&
                              data_spatial_shape.merge_rank(window_shape.rank()) &&
                              data_spatial_shape.merge_rank(window_strides.size()),
                          "Ranks for data item shape (data batch has shape ",
                          data_batch_shape,
                          ", so data item rank is ",
                          (data_batch_shape.rank() - 2),
                          "), padding below (",
                          data_padding_below,
                          "), padding above (",
                          data_padding_above,
                          "), window shape (",
                          window_shape,
                          "), and window strides (",
                          window_strides,
                          ") do not match.");

    Dimension batch_size{Dimension::dynamic()};
    Dimension channel_count{Dimension::dynamic()};
    ov::PartialShape data_output_spatial_shape{ov::PartialShape::dynamic(data_spatial_shape.rank())};

    if (data_batch_shape.rank().is_static()) {
        batch_size = data_batch_shape[0];
        channel_count = data_batch_shape[1];

        for (int64_t i = 0; i < data_spatial_shape.rank().get_length(); i++) {
            data_spatial_shape[i] = data_batch_shape[i + 2];
        }

        NODE_VALIDATION_CHECK(node, batch_size.is_dynamic() || batch_size.get_length() > 0, "Batch size is zero.");

        NODE_VALIDATION_CHECK(node,
                              channel_count.is_dynamic() || channel_count.get_length() > 0,
                              "Channel count is zero.");

        // For pooling ops we don't need dilation, so we fill in the identity value (all 1).
        Strides data_dilation(data_spatial_shape.rank().get_length(), 1);
        Strides dilations = window_dilation;
        // if the window_dilation was not specified, generate the default value (no dilations)
        if (window_dilation.empty()) {
            // dilations equal to 1 for each spatial axis mean that the window is not dilated
            dilations = Strides(data_spatial_shape.rank().get_length(), 1);
        }

        data_output_spatial_shape = infer_windowed_reduction_output_shape(node,
                                                                          data_spatial_shape,
                                                                          data_dilation,
                                                                          data_padding_below,
                                                                          data_padding_above,
                                                                          window_shape,
                                                                          window_strides,
                                                                          dilations,
                                                                          is_window_all_in_padding_allowed,
                                                                          ceil_mode);
    }

    ov::PartialShape data_batch_output_shape{ov::PartialShape::dynamic(data_output_spatial_shape.rank() + 2)};
    data_batch_output_shape[0] = batch_size;
    data_batch_output_shape[1] = channel_count;

    for (int64_t i = 0; i < data_spatial_shape.rank().get_length(); i++) {
        data_batch_output_shape[i + 2] = data_output_spatial_shape[i];
    }

    return data_batch_output_shape;
}

ov::PartialShape infer_slice_shape(const Node* node,
                                   const ov::PartialShape& input_shape,
                                   const std::vector<int64_t>& begin,
                                   const std::vector<int64_t>& end,
                                   const std::vector<int64_t>& strides,
                                   const ov::AxisSet& begin_mask,
                                   const ov::AxisSet& end_mask,
                                   const ov::AxisSet& new_axis_mask,
                                   const ov::AxisSet& shrink_axis_mask,
                                   const ov::AxisSet& ellipsis_mask) {
    if (begin.size() && end.size()) {
        NODE_VALIDATION_CHECK(node,
                              begin.size() == end.size(),
                              "Lower bounds and Upper bounds needs to have same number of values");
    }
    if (begin.size() && strides.size()) {
        NODE_VALIDATION_CHECK(node,
                              begin.size() == strides.size(),
                              "Lower bounds and strides needs to have same number of values");
    }
    if (end.size() && strides.size()) {
        NODE_VALIDATION_CHECK(node,
                              end.size() == strides.size(),
                              "Upper bounds and strides needs to have same number of values");
    }

    NODE_VALIDATION_CHECK(node, ellipsis_mask.size() <= 1, "At most one ellipsis is allowed.");

    if (input_shape.rank().is_dynamic()) {
        return ov::PartialShape::dynamic();
    }

    NODE_VALIDATION_CHECK(node,
                          input_shape.rank().get_length() + new_axis_mask.size() >= begin.size(),
                          "Input rank plus number of new axis has to be at least the size of Lower "
                          "and Upper bounds vector.");

    std::vector<Dimension> dim;

    int64_t input_shape_idx = 0;
    for (size_t axis = 0; axis < begin.size(); ++axis) {
        // add all dimensions hidden under the ellipsis mask if ellipsis mask is set
        if (ellipsis_mask.count(axis)) {
            // only one bit in ellipsis mask is allowed
            int num_new_axis_after_ellipses = 0;
            int num_input_axis_before_ellipses = 0;
            for (size_t i = 0; i < axis; ++i) {
                if (!new_axis_mask.count(i)) {
                    num_input_axis_before_ellipses++;
                }
            }
            for (size_t i = axis + 1; i < begin.size(); ++i) {
                if (new_axis_mask.count(i)) {
                    num_new_axis_after_ellipses++;
                }
            }

            int64_t num_input_axis_after_ellipses =
                (begin.size() - axis - num_new_axis_after_ellipses - 1);  // -1 because it's a position of ellipses
            int64_t num_of_hidden_dims =
                input_shape.rank().get_length() - num_input_axis_after_ellipses - num_input_axis_before_ellipses;
            for (int64_t i = 0; i < num_of_hidden_dims; ++i) {
                dim.emplace_back(input_shape[input_shape_idx]);
                input_shape_idx++;
            }
        } else {
            // add new single dimension if new_axis_mask is set
            if (new_axis_mask.count(axis)) {
                dim.emplace_back(1);
            }
            // skip this dimension if shrink_axis_mask is set
            else if (shrink_axis_mask.count(axis)) {
                input_shape_idx++;
            }
            // calculating dimension (begin, end, begin_mask, end_mask, stride)
            else {
                // check dynamic dimension
                if (input_shape[input_shape_idx].is_dynamic()) {
                    input_shape_idx++;
                    dim.emplace_back(Dimension::dynamic());
                    continue;
                }

                int64_t lb = begin[axis];
                int64_t ub = end[axis];

                // set default value for stride or use given value
                int64_t stride = 1;
                if (strides.size() > axis) {
                    stride = strides[axis];
                }
                NODE_VALIDATION_CHECK(node, stride != 0, "Stride must be non-zero");

                // convert negative indexes to positive
                // take max for this case: if abs(lb) > input_shape[input_shape_idx],then after
                // conversion lb < 0
                // so according to tensorflow and numpy we just get 0
                if (lb < 0) {
                    lb = std::max(input_shape[input_shape_idx].get_length() + lb, int64_t(0));
                }

                if (ub < 0) {
                    ub =
                        std::max(input_shape[input_shape_idx].get_length() + ub, stride > 0 ? int64_t(0) : int64_t(-1));
                }

                // apply restrictions when begin or end values more than max possible values.
                lb = std::min(input_shape[input_shape_idx].get_length(), lb);
                ub = std::min(input_shape[input_shape_idx].get_length(), ub);

                int64_t dimension = 0;
                if (stride < 0) {
                    // apply masks
                    if (begin_mask.count(axis)) {
                        lb = input_shape[input_shape_idx].get_length() - 1;
                    }
                    if (end_mask.count(axis)) {
                        ub = -1;
                    }

                    lb = std::min(lb, input_shape[input_shape_idx].get_length() - 1);
                    lb -= 1;  // we always get 1st element, so we need decrease range
                    if (ub <= lb) {
                        dimension = (ub - lb) / stride + 1;
                    }
                } else {
                    // apply masks
                    if (begin_mask.count(axis)) {
                        lb = 0;
                    }
                    if (end_mask.count(axis)) {
                        ub = input_shape[input_shape_idx].get_length();
                    }

                    lb += 1;  // we always get 1st element, so we need decrease range
                    if (ub >= lb) {
                        dimension = (ub - lb) / stride + 1;
                    }
                }

                dim.emplace_back(dimension);
                input_shape_idx++;
            }
        }
    }
    // get remaining values
    for (; input_shape_idx < input_shape.rank().get_length(); ++input_shape_idx) {
        dim.emplace_back(input_shape[input_shape_idx]);
    }

    return dim;
}

void opset1::infer_conv_backprop_auto_padding(const Shape& input_data_shape,
                                              const Shape& filters_shape,
                                              const Shape& output_shape,
                                              const Strides& strides,
                                              const Strides& dilations,
                                              const ov::op::PadType auto_pad_type,
                                              const CoordinateDiff& output_padding,
                                              CoordinateDiff& pads_begin,
                                              CoordinateDiff& pads_end) {
    OPENVINO_ASSERT(auto_pad_type == ov::op::PadType::SAME_UPPER || auto_pad_type == ov::op::PadType::SAME_LOWER);

    size_t num_spatial_dims = input_data_shape.size();
    OPENVINO_ASSERT(filters_shape.size() == num_spatial_dims && strides.size() == num_spatial_dims &&
                    dilations.size() == num_spatial_dims && pads_begin.size() == num_spatial_dims &&
                    pads_end.size() == num_spatial_dims && output_padding.size() == num_spatial_dims);

    pads_begin = CoordinateDiff(num_spatial_dims);
    pads_end = CoordinateDiff(num_spatial_dims);

    for (uint64_t i = 0; i < num_spatial_dims; ++i) {
        int total_padding = std::max<int>(
            static_cast<int>(strides[i] * (input_data_shape[i] - 1) + dilations[i] * (filters_shape[i] - 1) + 1 -
                             output_shape[i] + output_padding[i]),
            0);
        if (auto_pad_type != ov::op::PadType::SAME_UPPER) {
            pads_begin[i] = total_padding / 2;
            pads_end[i] = total_padding - pads_begin[i];
        } else {
            pads_end[i] = total_padding / 2;
            pads_begin[i] = total_padding - pads_end[i];
        }
    }
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
}  // namespace

std::shared_ptr<op::v0::Constant> get_constant_max_of_type(element::Type_t t) {
    auto tensor = ov::util::make_tensor_of_max_value(t);
    return tensor ? std::make_shared<op::v0::Constant>(tensor) : nullptr;
}

std::shared_ptr<op::v0::Constant> get_constant_min_of_type(element::Type_t t) {
    auto tensor = ov::util::make_tensor_of_min_value(t);
    return tensor ? std::make_shared<op::v0::Constant>(tensor) : nullptr;
}

std::shared_ptr<op::v0::Constant> get_constant_lowest_of_type(element::Type_t t) {
#define OPENVINO_TYPE_TO_LOWEST_CONST(t)                                                       \
    case t:                                                                                    \
        return op::v0::Constant::create(                                                       \
            t,                                                                                 \
            {},                                                                                \
            {std::numeric_limits<typename ov::element_type_traits<t>::value_type>::lowest()}); \
        break

    switch (t) {
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::boolean);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::bf16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::f16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::f32);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::f64);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i8);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i32);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::i64);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u1);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u8);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u16);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u32);
        OPENVINO_TYPE_TO_LOWEST_CONST(ov::element::u64);

    case ov::element::undefined:
    case ov::element::dynamic:
    default:
        return nullptr;
    }
}
}  // namespace ngraph

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

bool ov::util::are_unique(const std::vector<int64_t>& data) {
    return std::unordered_set<int64_t>(data.begin(), data.cend()).size() == data.size();
}

// clip value to min, max
int64_t ov::util::clip(const int64_t& value, const int64_t& min, const int64_t& max) {
    return std::min(std::max(value, min), max);
};

std::shared_ptr<ov::op::v0::Constant> ov::util::constantfold_subgraph(const ov::Output<Node>& subgraph_sink) {
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

namespace ov {
namespace util {
using ov::op::v0::Constant;

std::shared_ptr<Constant> get_constant_from_source(const ov::Output<Node>& source) {
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

std::vector<ov::PartialShape> get_tensors_partial_shapes(const TensorVector& tensors) {
    std::vector<ov::PartialShape> shapes;
    shapes.reserve(tensors.size());
    for (const auto& t : tensors) {
        shapes.emplace_back(t.get_shape());
    }
    return shapes;
}

std::vector<ov::PartialShape> get_node_input_partial_shapes(const Node& node) {
    std::vector<ov::PartialShape> shapes;
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

bool evaluate_as_partial_shape(const ov::Output<Node>& output, ov::PartialShape& pshape) {
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
        pshape = ov::PartialShape(resulting_pshape);
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
