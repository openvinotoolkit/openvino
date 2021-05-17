// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <vector>

#include "core/null_node.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "op/gather.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                std::vector<uint64_t>
                    get_normalized_axes_vector(const Node& onnx_node,
                                               const Rank& data_rank,
                                               const std::vector<int64_t> axes_attr)
                {
                    if (data_rank.is_static())
                    {
                        const auto normalized_axes_vec =
                            normalize_axes(onnx_node.get_description(), axes_attr, data_rank);
                        return std::vector<uint64_t>(std::begin(normalized_axes_vec),
                                                     std::end(normalized_axes_vec));
                    }
                    else
                    {
                        CHECK_VALID_NODE(onnx_node,
                                         std::all_of(std::begin(axes_attr),
                                                     std::end(axes_attr),
                                                     [](int64_t axis) { return axis >= 0; }),
                                         "All axes must be positive when data rank is unknown");
                        return std::vector<uint64_t>(std::begin(axes_attr), std::end(axes_attr));
                    }
                }

                /// \brief Transform Slice axes input to mask which is attribute of
                /// StridedSlice:v1 interface.
                ///
                /// \note Mask attributes of StridedSlice:10 operator indicates
                ///       if corresponding begin/end/strides input indices should be applied (0
                ///       value) or ignored (1 value)
                ///
                /// \param[in] axes                 Axes input of ONNX Slice operator
                /// \param[in] slice_indices_length Length of Slice indices
                ///                                 (starts, ends, steps)
                ///
                /// \return Mask attribute in format required by StridedSlice:v1
                std::vector<int64_t> axes_to_mask(const std::vector<uint64_t>& axes,
                                                  uint64_t slice_indices_length)
                {
                    std::vector<int64_t> mask(slice_indices_length, 1);
                    for (auto axis : axes)
                    {
                        mask[axis] = 0;
                    }
                    return mask;
                }

                /// \brief Adjsut ONNX Slice indices: starts, ends, steps to StridedSlice:v1
                /// interface.
                ///
                /// \note StridedSlice:v1 doesn't support axes paramets.
                ///       The axes parameters detrmines to which dimension of input data slice
                ///       operation should be applied.
                ///       The retuned sub-graph provide proper adjustement of Slice indices if
                ///       it is needed.
                ///
                /// \param[in] indices               Parameters of Slice operator: starts, ends,
                ///                                  steps.
                /// \param[in] axes                  Determines dimensions on which slice
                ///                                  operation should be applied.
                /// \param[in] slice_indices_length  Indices length after adjustment
                /// \param[in] fill_in_value         Neutral value (`0` for starts and ends,
                ///                                  `1` for steps) which is set to indices
                ///                                  in order to provide adjustment.
                ///
                /// \return Sub-graph represents adjusted indices or input indices
                ///         if any transformation was needed.
                Output<ngraph::Node> adjust_indices_if_needed(const Output<ngraph::Node>& indices,
                                                              const std::vector<uint64_t>& axes,
                                                              uint64_t slice_indices_length,
                                                              int64_t fill_in_value)
                {
                    const bool are_axes_sorted = std::is_sorted(axes.begin(), axes.end());

                    const auto indices_shape = indices.get_partial_shape();
                    // if length of slice indices vector is known
                    if (indices_shape.rank().is_static() &&
                        indices_shape.rank().get_length() == 1 && indices_shape[0].is_static())
                    {
                        if (static_cast<uint64_t>(indices_shape[0].get_length()) >=
                                slice_indices_length &&
                            are_axes_sorted)
                        {
                            // adjusting indices is not needed
                            return indices;
                        }
                    }
                    // Handle a case when starts/ends/steps lengths are less than provided axes
                    // in order to ensure compatibility with `StridedSlice:v1` interface
                    // Example:
                    // data_shape: {3, 3, 3, 3}
                    // starts: [1, 1] - after extending --> [0, 0, 1, 1]
                    // ends: [2, 2] - after extending --> [0, 0, 2, 2]
                    // steps : [0, 1] - after extending --> [1, 1, 0, 1] (`1` is neutral as a
                    // strides value)
                    // axes: [2, 3] - apply slice values to 2 and 3 dimension of input data
                    // expected_output_shape: {3, 3, 1, 1}
                    OutputVector adjusted_indices(slice_indices_length);
                    std::vector<uint64_t> target_axes(axes);
                    const auto gather_axis =
                        default_opset::Constant::create(indices.get_element_type(), {}, {0});

                    int added_indices_number = 0;
                    for (uint64_t i = 0; i < slice_indices_length; ++i)
                    {
                        if (std::find(std::begin(axes), std::end(axes), i) == axes.end())
                        {
                            adjusted_indices[i] = default_opset::Constant::create(
                                indices.get_element_type(), {1}, {fill_in_value});
                            target_axes.insert(std::next(target_axes.begin(), i), i);
                            ++added_indices_number;
                        }
                        else
                        {
                            adjusted_indices[i] = std::make_shared<default_opset::Gather>(
                                indices,
                                default_opset::Constant::create(
                                    indices.get_element_type(), {1}, {i - added_indices_number}),
                                gather_axis);
                        }
                    }

                    if (!are_axes_sorted)
                    {
                        OutputVector indices_tmp(adjusted_indices);
                        for (size_t i = 0; i < target_axes.size(); ++i)
                        {
                            adjusted_indices[target_axes[i]] = indices_tmp[i];
                        }
                    }

                    return std::make_shared<default_opset::Concat>(adjusted_indices, 0);
                }
            } // namespace

            namespace set_10
            {
                OutputVector slice(const Node& node)
                {
                    using ngraph::op::is_null;

                    OutputVector inputs{node.get_ng_inputs()};
                    const auto data = inputs.at(0);
                    const auto data_rank = data.get_partial_shape().rank();

                    auto starts = inputs.at(1);
                    auto ends = inputs.at(2);

                    // Slice is calculated over all axes as default
                    Output<ngraph::Node> axes;
                    if (inputs.size() >= 4 && !is_null(inputs.at(3))) // axes input provided
                    {
                        axes = inputs.at(3);
                        CHECK_VALID_NODE(node,
                                         ngraph::op::is_constant(axes.get_node()),
                                         "Axes input must be constant");
                    }
                    else
                    {
                        CHECK_VALID_NODE(
                            node,
                            data_rank.is_static(),
                            "Data rank must be static when axes input is not provided");
                        const size_t data_rank_value = data_rank.get_length();
                        axes = default_opset::Constant::create(
                            element::i64,
                            {data_rank_value},
                            common::get_monotonic_range<int64_t>(data_rank_value));
                    }

                    const auto axes_const =
                        as_type_ptr<default_opset::Constant>(axes.get_node_shared_ptr());
                    auto raw_axes_vec = axes_const->cast_vector<int64_t>();
                    std::vector<uint64_t> axes_vec =
                        get_normalized_axes_vector(node, data_rank, raw_axes_vec);

                    const size_t slice_indices_length =
                        *std::max_element(std::begin(axes_vec), std::end(axes_vec)) + 1;
                    const auto begin_end_mask = axes_to_mask(axes_vec, slice_indices_length);

                    Output<ngraph::Node> steps;
                    if (inputs.size() == 5 && !is_null(inputs.at(4))) // steps input provided
                    {
                        steps = inputs.at(4);
                    }
                    else
                    {
                        steps = default_opset::Constant::create(
                            element::i64,
                            {slice_indices_length},
                            std::vector<int64_t>(slice_indices_length, 1));
                    }

                    starts = adjust_indices_if_needed(starts, axes_vec, slice_indices_length, 0);
                    ends = adjust_indices_if_needed(ends, axes_vec, slice_indices_length, 0);
                    steps = adjust_indices_if_needed(steps, axes_vec, slice_indices_length, 1);

                    return {std::make_shared<default_opset::StridedSlice>(
                        data, starts, ends, steps, begin_end_mask, begin_end_mask)};
                }
            } // namespace set_10

            namespace set_1
            {
                OutputVector slice(const Node& node)
                {
                    Output<ngraph::Node> data = node.get_ng_inputs().at(0);
                    const auto data_rank = data.get_partial_shape().rank();

                    const auto starts_atr =
                        node.get_attribute_value<std::vector<int64_t>>("starts");
                    const auto ends_atr = node.get_attribute_value<std::vector<int64_t>>("ends");

                    std::shared_ptr<ngraph::Node> starts =
                        std::make_shared<default_opset::Constant>(
                            element::i64, Shape{starts_atr.size()}, starts_atr);
                    std::shared_ptr<ngraph::Node> ends = std::make_shared<default_opset::Constant>(
                        element::i64, Shape{ends_atr.size()}, ends_atr);

                    auto axes = node.get_attribute_value<std::vector<int64_t>>(
                        "axes", std::vector<int64_t>());

                    if (axes.empty())
                    {
                        CHECK_VALID_NODE(
                            node,
                            data_rank.is_static(),
                            "Data rank must be static when axes input is not provided");
                        axes = common::get_monotonic_range<int64_t>(data_rank.get_length());
                    }

                    std::vector<uint64_t> normalized_axes =
                        get_normalized_axes_vector(node, data_rank, axes);

                    const size_t slice_indices_length =
                        *std::max_element(std::begin(normalized_axes), std::end(normalized_axes)) +
                        1;
                    const auto begin_end_mask = axes_to_mask(normalized_axes, slice_indices_length);

                    std::shared_ptr<ngraph::Node> strides = default_opset::Constant::create(
                        element::i64,
                        Shape{slice_indices_length},
                        std::vector<int64_t>(slice_indices_length, 1));

                    starts =
                        adjust_indices_if_needed(starts, normalized_axes, slice_indices_length, 0)
                            .get_node_shared_ptr();
                    ends = adjust_indices_if_needed(ends, normalized_axes, slice_indices_length, 0)
                               .get_node_shared_ptr();
                    strides =
                        adjust_indices_if_needed(strides, normalized_axes, slice_indices_length, 1)
                            .get_node_shared_ptr();

                    return {std::make_shared<default_opset::StridedSlice>(
                        data, starts, ends, strides, begin_end_mask, begin_end_mask)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
