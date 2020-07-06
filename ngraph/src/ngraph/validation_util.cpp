//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>

#include "ngraph/evaluator.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

Strides ngraph::conv_default_strides(const Node* /* node */,
                                     const PartialShape& data_batch_shape,
                                     const PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && data_batch_shape.rank().get_length() >= 2)
    {
        rank = data_batch_shape.rank().get_length() - 2;
    }
    else if (filters_shape.rank().is_static() && filters_shape.rank().get_length() >= 2)
    {
        rank = filters_shape.rank().get_length() - 2;
    }
    else
    {
        rank = 0;
    }

    return Strides(rank, 1);
}

CoordinateDiff ngraph::conv_default_padding(const Node* /* node */,
                                            const PartialShape& data_batch_shape,
                                            const PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && data_batch_shape.rank().get_length() >= 2)
    {
        rank = data_batch_shape.rank().get_length() - 2;
    }
    else if (filters_shape.rank().is_static() && filters_shape.rank().get_length() >= 2)
    {
        rank = filters_shape.rank().get_length() - 2;
    }
    else
    {
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
PartialShape ngraph::infer_windowed_reduction_output_shape(const Node* node,
                                                           const PartialShape& data_shape,
                                                           const Strides& data_dilation,
                                                           const CoordinateDiff& data_padding_below,
                                                           const CoordinateDiff& data_padding_above,
                                                           const PartialShape& window_shape,
                                                           const Strides& window_strides,
                                                           const Strides& window_dilation,
                                                           bool is_window_all_in_padding_allowed,
                                                           bool ceil_mode)
{
    PartialShape data_shape_merged{PartialShape::dynamic()};

    NODE_VALIDATION_CHECK(node,
                          data_shape_merged.merge_rank(data_shape.rank()) &&
                              data_shape_merged.merge_rank(data_dilation.size()) &&
                              data_shape_merged.merge_rank(data_padding_below.size()) &&
                              data_shape_merged.merge_rank(data_padding_above.size()) &&
                              data_shape_merged.merge_rank(window_shape.rank()) &&
                              data_shape_merged.merge_rank(window_strides.size()) &&
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

    if (output_shape.rank().is_static())
    {
        for (size_t i = 0; i < output_shape.rank().get_length(); i++)
        {
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
            if (data_dim_static)
            {
                data_padded_dilated_dim =
                    (static_cast<int64_t>(data_dilation[i]) * (data_shape[i].get_length() - 1)) +
                    1 + data_padding_below[i] + data_padding_above[i];
                NODE_VALIDATION_CHECK(
                    node,
                    data_padded_dilated_dim > 0,
                    "Data shape after padding and dilation has dimension less than 1 (dim: ",
                    data_padded_dilated_dim,
                    ") at axis ",
                    i,
                    ".");
            }

            ptrdiff_t window_dilated_dim = -1;
            if (window_dim_static)
            {
                window_dilated_dim =
                    static_cast<int64_t>(window_dilation[i]) * (window_shape[i].get_length() - 1) +
                    1;

                NODE_VALIDATION_CHECK(node,
                                      window_dilated_dim > 0,
                                      "Window after dilation has dimension less than 1 (dim: ",
                                      window_dilated_dim,
                                      ") at axis ",
                                      i,
                                      ".");

                NODE_VALIDATION_CHECK(
                    node,
                    is_window_all_in_padding_allowed ||
                        (window_dilated_dim > data_padding_below[i] &&
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

            if (data_dim_static && window_dim_static)
            {
                NODE_VALIDATION_CHECK(node,
                                      window_dilated_dim <= data_padded_dilated_dim,
                                      "Window after dilation has dimension (dim: ",
                                      window_dilated_dim,
                                      ") larger than the data shape after padding (dim: ",
                                      data_padded_dilated_dim,
                                      ") at axis ",
                                      i,
                                      ".");

                if (ceil_mode)
                {
                    output_shape[i] = ceil_div(static_cast<size_t>(data_padded_dilated_dim) -
                                                   static_cast<size_t>(window_dilated_dim),
                                               window_strides[i]) +
                                      1;
                }
                else
                {
                    output_shape[i] = ((static_cast<size_t>(data_padded_dilated_dim) -
                                        static_cast<size_t>(window_dilated_dim)) /
                                       window_strides[i]) +
                                      1;
                }
            }
        }
    }

    return output_shape;
}

//
// Infers the output batch shape and element type for convolution fprop.
//
PartialShape ngraph::infer_convolution_forward(const Node* node,
                                               const PartialShape& data_batch_shape,
                                               const Strides& data_dilation,
                                               const CoordinateDiff& data_padding_below,
                                               const CoordinateDiff& data_padding_above,
                                               const PartialShape& filters_shape,
                                               const Strides& filter_strides,
                                               const Strides& filter_dilation)
{
    Rank data_batch_filters_rank{Rank::dynamic()};

    NODE_VALIDATION_CHECK(
        node,
        Rank::merge(data_batch_filters_rank, data_batch_shape.rank(), filters_shape.rank()),
        "Data batch and filters rank do not match (data batch shape: ",
        data_batch_shape,
        ", filters shape: ",
        filters_shape,
        ").");

    NODE_VALIDATION_CHECK(node,
                          data_batch_filters_rank.is_dynamic() ||
                              data_batch_filters_rank.get_length() >= 3,
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

    Dimension batch_size =
        (data_batch_shape.rank().is_static() ? data_batch_shape[0] : Dimension::dynamic());
    Dimension data_channel_count =
        (data_batch_shape.rank().is_static() ? data_batch_shape[1] : Dimension::dynamic());
    PartialShape data_spatial_shape(PartialShape::dynamic(spatial_rank));

    Dimension filter_output_channel_count =
        (filters_shape.rank().is_static() ? filters_shape[0] : Dimension::dynamic());
    Dimension filter_input_channel_count =
        (filters_shape.rank().is_static() ? filters_shape[1] : Dimension::dynamic());
    PartialShape filter_spatial_shape(PartialShape::dynamic(spatial_rank));

    //
    // Note: spatial_rank is definitely static at this point.
    //

    for (size_t i = 0; i < spatial_rank.get_length(); i++)
    {
        if (data_batch_shape.rank().is_static())
        {
            data_spatial_shape[i] = data_batch_shape[i + 2];
        }

        if (filters_shape.rank().is_static())
        {
            filter_spatial_shape[i] = filters_shape[i + 2];
        }
    }

    NODE_VALIDATION_CHECK(
        node, batch_size.is_dynamic() || batch_size.get_length() > 0, "Batch size is zero.");

    Dimension merged_channel_count;

    NODE_VALIDATION_CHECK(
        node,
        Dimension::merge(merged_channel_count, data_channel_count, filter_input_channel_count),
        "Data batch channel count (",
        data_channel_count,
        ") does not match filter input ",
        "channel count (",
        filter_input_channel_count,
        ").");

    NODE_VALIDATION_CHECK(node,
                          merged_channel_count.is_dynamic() ||
                              merged_channel_count.get_length() > 0,
                          "Data batch channel count and/or filter input channel count is zero.");

    NODE_VALIDATION_CHECK(node,
                          filter_output_channel_count.is_dynamic() ||
                              filter_output_channel_count.get_length() > 0,
                          "Filter output channel count is zero.");

    PartialShape data_output_shape = infer_windowed_reduction_output_shape(node,
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

    for (size_t i = 0; i < spatial_rank.get_length(); i++)
    {
        batch_output_shape[i + 2] = data_output_shape[i];
    }

    return batch_output_shape;
}

//
// Infers the output batch shape and element type for batched pooling fprop.
//
PartialShape ngraph::infer_batched_pooling_forward(const Node* node,
                                                   const PartialShape& data_batch_shape,
                                                   const CoordinateDiff& data_padding_below,
                                                   const CoordinateDiff& data_padding_above,
                                                   const PartialShape& window_shape,
                                                   const Strides& window_strides,
                                                   bool is_window_all_in_padding_allowed,
                                                   bool ceil_mode)
{
    NODE_VALIDATION_CHECK(node,
                          data_batch_shape.rank().is_dynamic() ||
                              data_batch_shape.rank().get_length() >= 3,
                          "Data batch must have rank of at least 3 (one batch axis, ",
                          "one input-channel axis, and at least one spatial dimension) ",
                          "(data batch shape: ",
                          data_batch_shape,
                          ").");

    PartialShape data_spatial_shape{PartialShape::dynamic()};

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
    PartialShape data_output_spatial_shape{PartialShape::dynamic(data_spatial_shape.rank())};

    if (data_batch_shape.rank().is_static())
    {
        batch_size = data_batch_shape[0];
        channel_count = data_batch_shape[1];

        for (size_t i = 0; i < data_spatial_shape.rank().get_length(); i++)
        {
            data_spatial_shape[i] = data_batch_shape[i + 2];
        }

        NODE_VALIDATION_CHECK(
            node, batch_size.is_dynamic() || batch_size.get_length() > 0, "Batch size is zero.");

        NODE_VALIDATION_CHECK(node,
                              channel_count.is_dynamic() || channel_count.get_length() > 0,
                              "Channel count is zero.");

        // For pooling ops we don't need dilation, so we fill in the identity value (all 1).
        Strides data_dilation(data_spatial_shape.rank().get_length(), 1);
        Strides window_dilation(data_spatial_shape.rank().get_length(), 1);

        data_output_spatial_shape =
            infer_windowed_reduction_output_shape(node,
                                                  data_spatial_shape,
                                                  data_dilation,
                                                  data_padding_below,
                                                  data_padding_above,
                                                  window_shape,
                                                  window_strides,
                                                  window_dilation,
                                                  is_window_all_in_padding_allowed,
                                                  ceil_mode);
    }

    PartialShape data_batch_output_shape{
        PartialShape::dynamic(data_output_spatial_shape.rank() + 2)};
    data_batch_output_shape[0] = batch_size;
    data_batch_output_shape[1] = channel_count;

    for (size_t i = 0; i < data_spatial_shape.rank().get_length(); i++)
    {
        data_batch_output_shape[i + 2] = data_output_spatial_shape[i];
    }

    return data_batch_output_shape;
}

struct ChannelShapedInputSpec
{
    element::Type m_element_type;
    PartialShape m_shape;
    std::string m_input_name;
};

static std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward_helper(
    const Node* node,
    element::Type input_element_type,
    const PartialShape& input_shape,
    const std::vector<ChannelShapedInputSpec>& channel_shaped_inputs)
{
    // Built up a slash-separated string naming all the channel-shaped inputs, for use in error
    // messages.
    std::stringstream ss;
    bool first = true;
    for (auto& inp : channel_shaped_inputs)
    {
        if (!first)
        {
            ss << "/";
        }
        ss << inp.m_input_name;
        first = false;
    }
    std::string channel_input_names = ss.str();

    // Infer output element type.
    element::Type et_result{input_element_type};

    for (auto& inp : channel_shaped_inputs)
    {
        NODE_VALIDATION_CHECK(node,
                              element::Type::merge(et_result, et_result, inp.m_element_type),
                              "Input element types do not match.");
    }

    // Extract channel dimension from input shape.
    Dimension channel_dim{Dimension::dynamic()};

    NODE_VALIDATION_CHECK(node,
                          input_shape.is_dynamic() || input_shape.rank().get_length() >= 2,
                          "Input argument must have rank of at least 2 (input argument shape: ",
                          input_shape,
                          ").");

    if (input_shape.rank().is_static())
    {
        channel_dim = input_shape[1];
    }

    // Infer gamma/beta/mu/sigma shape, which must be consistent with a vector of size
    // "channel_dim".
    PartialShape channel_shape{PartialShape::dynamic()};

    for (auto& inp : channel_shaped_inputs)
    {
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

    if (batch_result_shape.rank().is_static())
    {
        batch_result_shape[1] = channel_dim;
    }

    return std::make_tuple(et_result, batch_result_shape, PartialShape{channel_dim});
}

std::tuple<element::Type, PartialShape, PartialShape>
    ngraph::infer_batch_norm_forward(const Node* node,
                                     element::Type input_element_type,
                                     element::Type gamma_element_type,
                                     element::Type beta_element_type,
                                     element::Type mean_element_type,
                                     element::Type variance_element_type,
                                     const PartialShape& input_shape,
                                     const PartialShape& gamma_shape,
                                     const PartialShape& beta_shape,
                                     const PartialShape& mean_shape,
                                     const PartialShape& variance_shape)
{
    return infer_batch_norm_forward_helper(node,
                                           input_element_type,
                                           input_shape,
                                           {{gamma_element_type, gamma_shape, "gamma"},
                                            {beta_element_type, beta_shape, "beta"},
                                            {mean_element_type, mean_shape, "mean"},
                                            {variance_element_type, variance_shape, "variance"}});
}

std::tuple<element::Type, PartialShape, PartialShape>
    ngraph::infer_batch_norm_forward(const Node* node,
                                     element::Type input_element_type,
                                     element::Type gamma_element_type,
                                     element::Type beta_element_type,
                                     const PartialShape& input_shape,
                                     const PartialShape& gamma_shape,
                                     const PartialShape& beta_shape)
{
    return infer_batch_norm_forward_helper(
        node,
        input_element_type,
        input_shape,
        {{gamma_element_type, gamma_shape, "gamma"}, {beta_element_type, beta_shape, "beta"}});
}

void ngraph::infer_auto_padding(const Shape& image_shape,
                                const Shape& filter_shape,
                                const Strides& filter_strides,
                                const Strides& filter_dilations,
                                const op::PadType pad_type,
                                CoordinateDiff& padding_above,
                                CoordinateDiff& padding_below)
{
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

bool ngraph::try_apply_auto_padding(const PartialShape& image_shape,
                                    const Shape& filter_shape,
                                    const Strides& filter_strides,
                                    const Strides& filter_dilations,
                                    const op::PadType pad_type,
                                    CoordinateDiff& padding_above,
                                    CoordinateDiff& padding_below)
{
    NGRAPH_CHECK(pad_type == op::PadType::SAME_UPPER || pad_type == op::PadType::SAME_LOWER);

    if (image_shape.rank().is_dynamic())
    {
        return false;
    }
    const auto image_dims = static_cast<std::vector<Dimension>>(image_shape);
    const bool are_spatial_dims_static =
        std::all_of(std::begin(image_dims) + 2, std::end(image_dims), [](const Dimension& dim) {
            return dim.is_static();
        });
    if (!are_spatial_dims_static)
    {
        return false;
    }

    for (size_t i = 0; i < static_cast<size_t>(filter_shape.size()); i++)
    {
        int64_t image_size = static_cast<int64_t>(image_dims[i + 2]);
        int64_t filter_size = (static_cast<int64_t>(filter_shape[i]) - 1) * filter_dilations[i] + 1;
        int64_t filter_stride = static_cast<int64_t>(filter_strides[i]);
        auto output_size = (image_size + filter_stride - 1) / filter_stride;

        auto padding_needed =
            std::max(int64_t(0), (output_size - 1) * filter_stride + filter_size - image_size);
        auto padding_lhs = padding_needed / 2;
        auto padding_rhs = padding_needed - padding_lhs;
        padding_below.push_back(pad_type == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs);
        padding_above.push_back(pad_type == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs);
    }
    return true;
}

PartialShape ngraph::infer_slice_shape(const Node* node,
                                       const PartialShape& input_shape,
                                       const std::vector<int64_t>& begin,
                                       const std::vector<int64_t>& end,
                                       const std::vector<int64_t>& strides,
                                       const AxisSet& begin_mask,
                                       const AxisSet& end_mask,
                                       const AxisSet& new_axis_mask,
                                       const AxisSet& shrink_axis_mask,
                                       const AxisSet& ellipsis_mask)
{
    if (begin.size() && end.size())
    {
        NODE_VALIDATION_CHECK(node,
                              begin.size() == end.size(),
                              "Lower bounds and Upper bounds needs to have same number of values");
    }
    if (begin.size() && strides.size())
    {
        NODE_VALIDATION_CHECK(node,
                              begin.size() == strides.size(),
                              "Lower bounds and strides needs to have same number of values");
    }
    if (end.size() && strides.size())
    {
        NODE_VALIDATION_CHECK(node,
                              end.size() == strides.size(),
                              "Upper bounds and strides needs to have same number of values");
    }

    NODE_VALIDATION_CHECK(node, ellipsis_mask.size() <= 1, "At most one ellipsis is allowed.");

    if (input_shape.rank().is_dynamic())
    {
        return PartialShape::dynamic();
    }

    NODE_VALIDATION_CHECK(node,
                          input_shape.rank().get_length() + new_axis_mask.size() >= begin.size(),
                          "Input rank plus number of new axis has to be at least the size of Lower "
                          "and Upper bounds vector.");

    std::vector<Dimension> dim;

    size_t input_shape_idx = 0;
    for (size_t axis = 0; axis < begin.size(); ++axis)
    {
        // add all dimensions hidden under the ellipsis mask if ellipsis mask is set
        if (ellipsis_mask.count(axis))
        {
            // only one bit in ellipsis mask is allowed
            int num_new_axis_after_ellipses = 0;
            int num_input_axis_before_ellipses = 0;
            for (size_t i = 0; i < axis; ++i)
            {
                if (!new_axis_mask.count(i))
                {
                    num_input_axis_before_ellipses++;
                }
            }
            for (size_t i = axis + 1; i < begin.size(); ++i)
            {
                if (new_axis_mask.count(i))
                {
                    num_new_axis_after_ellipses++;
                }
            }

            int64_t num_input_axis_after_ellipses =
                (begin.size() - axis - num_new_axis_after_ellipses -
                 1); // -1 because it's a position of ellipses
            int64_t num_of_hidden_dims = input_shape.rank().get_length() -
                                         num_input_axis_after_ellipses -
                                         num_input_axis_before_ellipses;
            for (int64_t i = 0; i < num_of_hidden_dims; ++i)
            {
                dim.emplace_back(input_shape[input_shape_idx]);
                input_shape_idx++;
            }
        }
        else
        {
            // add new single dimension if new_axis_mask is set
            if (new_axis_mask.count(axis))
            {
                dim.emplace_back(1);
            }
            // skip this dimension if shrink_axis_mask is set
            else if (shrink_axis_mask.count(axis))
            {
                input_shape_idx++;
            }
            // calculating dimension (begin, end, begin_mask, end_mask, stride)
            else
            {
                // check dynamic dimension
                if (input_shape[input_shape_idx].is_dynamic())
                {
                    input_shape_idx++;
                    dim.emplace_back(Dimension::dynamic());
                    continue;
                }

                int64_t lb = begin[axis];
                int64_t ub = end[axis];

                // convert negative indexes to positive
                // take max for this case: if abs(lb) > input_shape[input_shape_idx],then after
                // conversion lb < 0
                // so according to tensorflow and numpy we just get 0
                if (lb < 0)
                {
                    lb = std::max(input_shape[input_shape_idx].get_length() + lb, int64_t(0));
                }

                if (ub < 0)
                {
                    ub = std::max(input_shape[input_shape_idx].get_length() + ub, int64_t(0));
                }

                // apply restrictions when begin or end values more than max possible values.
                lb = std::min(input_shape[input_shape_idx].get_length(), lb);
                ub = std::min(input_shape[input_shape_idx].get_length(), ub);

                // set default value for stride or use given value
                int64_t stride = 1;
                if (strides.size() > axis)
                {
                    stride = strides[axis];
                }

                NODE_VALIDATION_CHECK(node, stride != 0, "Stride must be non-zero");

                int64_t dimension = 0;
                if (stride < 0)
                {
                    // apply masks
                    if (begin_mask.count(axis))
                    {
                        lb = input_shape[input_shape_idx].get_length() - 1;
                    }
                    if (end_mask.count(axis))
                    {
                        ub = -1;
                    }

                    lb = std::min(lb, input_shape[input_shape_idx].get_length() - 1);
                    lb -= 1; // we always get 1st element, so we need decrease range
                    if (ub <= lb)
                    {
                        dimension = (ub - lb) / stride + 1;
                    }
                }
                else
                {
                    // apply masks
                    if (begin_mask.count(axis))
                    {
                        lb = 0;
                    }
                    if (end_mask.count(axis))
                    {
                        ub = input_shape[input_shape_idx].get_length();
                    }

                    lb += 1; // we always get 1st element, so we need decrease range
                    if (ub >= lb)
                    {
                        dimension = (ub - lb) / stride + 1;
                    }
                }

                dim.emplace_back(dimension);
                input_shape_idx++;
            }
        }
    }
    // get remaining values
    for (; input_shape_idx < input_shape.rank().get_length(); ++input_shape_idx)
    {
        dim.emplace_back(input_shape[input_shape_idx]);
    }

    return dim;
}

std::vector<size_t> ngraph::normalize_axes(const std::string& node_description,
                                           const std::vector<int64_t>& axes,
                                           const Rank& tensor_rank)
{
    std::vector<size_t> new_axes;

    for (const auto& axis : axes)
    {
        new_axes.push_back(normalize_axis(node_description, axis, tensor_rank));
    }

    return new_axes;
}

int64_t ngraph::normalize_axis(const Node* node, std::int64_t axis, const Rank& tensor_rank)
{
    return normalize_axis(node->description(), axis, tensor_rank);
}

int64_t ngraph::normalize_axis(const std::string& node_description,
                               std::int64_t axis,
                               const Rank& tensor_rank)
{
    if (axis < 0)
    {
        // Handling negative axis requires static tensor rank
        NGRAPH_CHECK(tensor_rank.is_static(),
                     node_description,
                     " Rank must be static in order to normalize negative axis=",
                     axis);
    }
    if (tensor_rank.is_dynamic())
    {
        return axis;
    }

    const auto tensor_rank_value = tensor_rank.get_length();
    return normalize_axis(node_description,
                          axis,
                          tensor_rank_value,
                          -tensor_rank_value,
                          tensor_rank_value ? (tensor_rank_value - 1) : 0);
}

int64_t ngraph::normalize_axis(const Node* node,
                               std::int64_t axis,
                               std::uint64_t tensor_rank,
                               std::int64_t axis_range_min,
                               std::int64_t axis_range_max)
{
    return ngraph::normalize_axis(
        node->description(), axis, tensor_rank, axis_range_min, axis_range_max);
}

int64_t ngraph::normalize_axis(const std::string& node_description,
                               std::int64_t axis,
                               std::uint64_t tensor_rank,
                               std::int64_t axis_range_min,
                               std::int64_t axis_range_max)
{
    // Accepted range of value for axis is [axis_range_min, axis_range_max].
    NGRAPH_CHECK(((axis >= axis_range_min) && (axis <= axis_range_max)),
                 node_description,
                 " Parameter axis ",
                 axis,
                 " out of the tensor rank range [",
                 axis_range_min,
                 ", ",
                 axis_range_max,
                 "].");

    if (axis < 0)
    {
        axis = axis + tensor_rank;
    }

    return int64_t(axis);
}

void ngraph::opset1::infer_conv_backprop_auto_padding(const Shape& input_data_shape,
                                                      const Shape& filters_shape,
                                                      const Shape& output_shape,
                                                      const Strides& strides,
                                                      const Strides& dilations,
                                                      const op::PadType auto_pad_type,
                                                      const CoordinateDiff& output_padding,
                                                      CoordinateDiff& pads_begin,
                                                      CoordinateDiff& pads_end)
{
    NGRAPH_CHECK(auto_pad_type == op::PadType::SAME_UPPER ||
                 auto_pad_type == op::PadType::SAME_LOWER);

    size_t num_spatial_dims = input_data_shape.size();
    NGRAPH_CHECK(filters_shape.size() == num_spatial_dims && strides.size() == num_spatial_dims &&
                 dilations.size() == num_spatial_dims && pads_begin.size() == num_spatial_dims &&
                 pads_end.size() == num_spatial_dims && output_padding.size() == num_spatial_dims);

    pads_begin = CoordinateDiff(num_spatial_dims);
    pads_end = CoordinateDiff(num_spatial_dims);

    for (uint64_t i = 0; i < num_spatial_dims; ++i)
    {
        int total_padding = strides[i] * (input_data_shape[i] - 1) +
                            dilations[i] * (filters_shape[i] - 1) + 1 - output_shape[i] +
                            output_padding[i];
        if (auto_pad_type != op::PadType::SAME_UPPER)
        {
            pads_begin[i] = total_padding / 2;
            pads_end[i] = total_padding - pads_begin[i];
        }
        else
        {
            pads_end[i] = total_padding / 2;
            pads_begin[i] = total_padding - pads_end[i];
        }
    }
}

namespace
{
    /// \brief Scalar variant describes value of an Output, for use in max shape determination
    ///
    /// For tensor values, we use the maximum value in the tensor
    struct MaxValue
    {
        /// \brief No information known about the output
        MaxValue() {}
        /// \brief uint64_t assoiated with the output
        MaxValue(uint64_t value)
            : m_value(value)
        {
        }
        MaxValue(const vector<uint64_t>& slices, int64_t slice_axis)
            : m_slices(slices)
            , m_slice_axis(slice_axis)
        {
            m_value = *max_element(m_slices.begin(), m_slices.end());
        }
        uint64_t m_value{numeric_limits<uint64_t>::max()};
        vector<uint64_t> m_slices;
        int64_t m_slice_axis{-1};
    };

    vector<MaxValue> exec_constant(Node* node, vector<MaxValue>& inputs)
    {
        auto result = MaxValue();
        auto op = as_type<op::Constant>(node);
        auto element_type = op->get_output_element_type(0);
        if (element_type.is_integral())
        {
            uint64_t max_val = 0;
            if (element_type.is_signed())
            {
                for (auto elt : op->cast_vector<int64_t>())
                {
                    if (max_val < elt)
                    {
                        max_val = elt;
                    }
                }
            }
            else
            {
                for (auto elt : op->cast_vector<uint64_t>())
                {
                    if (max_val < elt)
                    {
                        max_val = elt;
                    }
                }
            }
            result = MaxValue(max_val);
        }
        return {result};
    }

    vector<MaxValue> exec_minimum(Node* node, vector<MaxValue>& inputs)
    {
        uint64_t min_value = numeric_limits<uint64_t>::max();
        switch (node->get_output_element_type(0))
        {
        case element::Type_t::i8: min_value = numeric_limits<int8_t>::max(); break;
        case element::Type_t::i16: min_value = numeric_limits<int16_t>::max(); break;
        case element::Type_t::i32: min_value = numeric_limits<int32_t>::max(); break;
        case element::Type_t::i64: min_value = numeric_limits<int64_t>::max(); break;
        case element::Type_t::u8: min_value = numeric_limits<uint8_t>::max(); break;
        case element::Type_t::u16: min_value = numeric_limits<uint16_t>::max(); break;
        case element::Type_t::u32: min_value = numeric_limits<uint32_t>::max(); break;
        case element::Type_t::u64: min_value = numeric_limits<uint64_t>::max(); break;
        default: break;
        }
        min_value = min(min_value, inputs.at(0).m_value);
        min_value = min(min_value, inputs.at(1).m_value);
        return {MaxValue(min_value)};
    }

    vector<MaxValue> exec_concat(Node* node, vector<MaxValue>& inputs)
    {
        auto op = as_type<op::v0::Concat>(node);
        vector<uint64_t> slice_maxen;
        for (auto input : inputs)
        {
            slice_maxen.push_back(input.m_value);
        }
        auto axis = op->get_concatenation_axis();
        return {MaxValue(slice_maxen, axis)};
    }

    vector<MaxValue> exec_reduce_min(Node* node, vector<MaxValue>& inputs)
    {
        auto data = inputs.at(0);
        if (data.m_slice_axis >= 0 && data.m_slices.size() > 1)
        {
            if (auto indices_const = as_type<op::v0::Constant>(node->get_input_node_ptr(1)))
            {
                if (indices_const->get_output_element_type(0).is_integral())
                {
                    auto indices_shape = indices_const->get_output_shape(0);
                    if (indices_shape == Shape{1})
                    {
                        auto indices = indices_const->cast_vector<int64_t>();
                        auto axis = indices.at(0);
                        if (axis == data.m_slice_axis)
                        {
                            return {
                                MaxValue(*min_element(data.m_slices.begin(), data.m_slices.end()))};
                        }
                    }
                }
            }
        }
        // Noting we can do
        return {MaxValue(data.m_value)};
    }
    vector<MaxValue> exec_nop(Node* node, vector<MaxValue>& inputs) { return {inputs.at(0)}; }
}

pair<bool, uint64_t> ngraph::maximum_value(const Output<Node>& value)
{
    static Evaluator<MaxValue>::op_handler_map handlers = {
        {op::v0::Concat::type_info, exec_concat},
        {op::v0::Constant::type_info, exec_constant},
        {op::v0::Convert::type_info, exec_nop},
        {op::v0::Minimum::type_info, exec_minimum},
        {op::v1::Minimum::type_info, exec_minimum},
        {op::v1::ReduceMin::type_info, exec_reduce_min},
        {op::v0::Squeeze::type_info, exec_nop},
        {op::v0::Unsqueeze::type_info, exec_nop}};
    Evaluator<MaxValue>::value_map value_map;
    Evaluator<MaxValue> evaluator(handlers, value_map);
    auto val = evaluator.evaluate(value);
    return pair<bool, uint64_t>(val.m_value < numeric_limits<uint64_t>::max(), val.m_value);
}

void ngraph::evaluate_nodes(std::map<RawNodeOutput, HostTensorPtr>& value_map,
                            std::map<RawNodeOutput, HostTensorPtr>& output_tensor_map,
                            const OutputVector& outputs)
{
    Evaluator<HostTensorPtr> evaluator({}, value_map);
    evaluator.set_univeral_handler(
        [&output_tensor_map](Node* node,
                             const HostTensorVector& input_tensors) -> HostTensorVector {
            HostTensorVector output_tensors;
            for (auto v : node->outputs())
            {
                auto it = output_tensor_map.find(v);
                if (it == output_tensor_map.end())
                {
                    auto c = make_shared<HostTensor>(v);
                    output_tensors.push_back(c);
                }
                else
                {
                    output_tensors.push_back(it->second);
                }
            }
            if (node->evaluate(output_tensors, input_tensors))
            {
                return output_tensors;
            }
            else
            {
                NGRAPH_CHECK(false, "Evaluation failed on ", node);
            }
        });
    for (auto value : outputs)
    {
        evaluator.evaluate(value);
    }
}
