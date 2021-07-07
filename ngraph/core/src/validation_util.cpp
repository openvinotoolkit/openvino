// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <ngraph/ops.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>

#include "ngraph/evaluator.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"
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
        for (int64_t i = 0; i < output_shape.rank().get_length(); i++)
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

void ngraph::validate_conv_params_spatial_dimensions(const Node* node,
                                                     const size_t num_spatial_dims,
                                                     const op::PadType auto_pad,
                                                     Strides& strides,
                                                     Strides& dilations,
                                                     CoordinateDiff& pads_begin,
                                                     CoordinateDiff& pads_end)
{
    if (strides.size() == 0)
    {
        strides = Strides(num_spatial_dims, 1);
    }
    if (dilations.size() == 0)
    {
        dilations = Strides(num_spatial_dims, 1);
    }
    if (pads_begin.size() == 0 || auto_pad == op::PadType::VALID)
    {
        pads_begin = CoordinateDiff(num_spatial_dims, 0);
    }
    if (pads_end.size() == 0 || auto_pad == op::PadType::VALID)
    {
        pads_end = CoordinateDiff(num_spatial_dims, 0);
    }
    NODE_VALIDATION_CHECK(node,
                          strides.size() == num_spatial_dims,
                          "Strides should be defined for all and only spatial features.");
    NODE_VALIDATION_CHECK(node,
                          dilations.size() == num_spatial_dims,
                          "Dilations should be defined for all and only spatial features.");
    NODE_VALIDATION_CHECK(node,
                          pads_begin.size() == num_spatial_dims &&
                              pads_end.size() == num_spatial_dims,
                          "Pads should be defined for all and only spatial features.");
}

PartialShape ngraph::validate_and_infer_convolution_forward_output_shape(
    const Node* node,
    const Rank& result_ps_rank,
    const PartialShape& data_batch_pshape,
    const PartialShape& filters_pshape,
    const op::PadType auto_pad,
    Strides& strides,
    Strides& dilations,
    CoordinateDiff& pads_begin,
    CoordinateDiff& pads_end)
{
    PartialShape result_shape = PartialShape::dynamic();
    if (result_ps_rank.is_static())
    {
        const auto num_spatial_dims = result_ps_rank.get_length() - 2;
        validate_conv_params_spatial_dimensions(
            node, num_spatial_dims, auto_pad, strides, dilations, pads_begin, pads_end);

        result_shape = PartialShape::dynamic(result_ps_rank);
        if (data_batch_pshape.rank().is_static())
        {
            result_shape[0] = data_batch_pshape[0]; // batch size
        }
        if (filters_pshape.rank().is_static())
        {
            result_shape[1] = filters_pshape[0]; // filter channel size
        }
        if (auto_pad == op::PadType::SAME_UPPER || auto_pad == op::PadType::SAME_LOWER)
        {
            bool auto_padding_applied = false;
            if (filters_pshape.rank().is_static() && filters_pshape.rank().get_length() > 2)
            {
                pads_begin.clear();
                pads_end.clear();

                const PartialShape filter_spatial_shape = [filters_pshape]() {
                    vector<Dimension> filter_dims{filters_pshape};
                    filter_dims.erase(filter_dims.begin(),
                                      filter_dims.begin() + 2); // Remove {C_OUT, C_IN}
                    return PartialShape{filter_dims};
                }();

                if (filter_spatial_shape.is_static())
                {
                    auto_padding_applied = try_apply_auto_padding(data_batch_pshape,
                                                                  filter_spatial_shape.to_shape(),
                                                                  strides,
                                                                  dilations,
                                                                  auto_pad,
                                                                  pads_end,
                                                                  pads_begin);
                }
            }
            if (!auto_padding_applied)
            {
                return result_shape;
            }
        }
        result_shape =
            infer_convolution_forward(node,
                                      data_batch_pshape,
                                      Strides(num_spatial_dims, 1), // dummy data dilations
                                      pads_begin,
                                      pads_end,
                                      filters_pshape,
                                      strides,
                                      dilations);
    }
    return result_shape;
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

    for (int64_t i = 0; i < spatial_rank.get_length(); i++)
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

    for (int64_t i = 0; i < spatial_rank.get_length(); i++)
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
                              (data_batch_shape.rank().get_length() >= 3 &&
                               data_batch_shape.rank().get_length() <= 5),
                          "Data batch must have rank of at least 4 or 5 (one batch axis, ",
                          "one input-channel axis, and two or three spatial dimension) ",
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

        for (int64_t i = 0; i < data_spatial_shape.rank().get_length(); i++)
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

    for (int64_t i = 0; i < data_spatial_shape.rank().get_length(); i++)
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
    for (const auto& inp : channel_shaped_inputs)
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

    for (const auto& inp : channel_shaped_inputs)
    {
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
    if (input_rank.is_static())
    {
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

    for (const auto& inp : channel_shaped_inputs)
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
    for (size_t i = 0; i < static_cast<size_t>(filter_shape.size()); i++)
    {
        if (image_dims[i + 2].is_static())
        {
            int64_t image_size = static_cast<int64_t>(image_dims[i + 2].get_length());
            int64_t filter_size =
                (static_cast<int64_t>(filter_shape[i]) - 1) * filter_dilations[i] + 1;
            int64_t filter_stride = static_cast<int64_t>(filter_strides[i]);
            auto output_size = (image_size + filter_stride - 1) / filter_stride;

            auto padding_needed =
                std::max(int64_t(0), (output_size - 1) * filter_stride + filter_size - image_size);
            auto padding_lhs = padding_needed / 2;
            auto padding_rhs = padding_needed - padding_lhs;
            padding_below.push_back(pad_type == op::PadType::SAME_UPPER ? padding_lhs
                                                                        : padding_rhs);
            padding_above.push_back(pad_type == op::PadType::SAME_UPPER ? padding_rhs
                                                                        : padding_lhs);
        }
        else
        {
            padding_below.push_back(0);
            padding_above.push_back(0);
        }
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

    int64_t input_shape_idx = 0;
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

                // set default value for stride or use given value
                int64_t stride = 1;
                if (strides.size() > axis)
                {
                    stride = strides[axis];
                }
                NODE_VALIDATION_CHECK(node, stride != 0, "Stride must be non-zero");

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
                    ub = std::max(input_shape[input_shape_idx].get_length() + ub,
                                  stride > 0 ? int64_t(0) : int64_t(-1));
                }

                // apply restrictions when begin or end values more than max possible values.
                lb = std::min(input_shape[input_shape_idx].get_length(), lb);
                ub = std::min(input_shape[input_shape_idx].get_length(), ub);

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
        int total_padding = std::max<int>(strides[i] * (input_data_shape[i] - 1) +
                                              dilations[i] * (filters_shape[i] - 1) + 1 -
                                              output_shape[i] + output_padding[i],
                                          0);
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
                    if (max_val < static_cast<uint64_t>(elt))
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

    vector<MaxValue> exec_shape_of(Node* node, vector<MaxValue>& inputs)
    {
        const auto& inputPS = node->get_input_partial_shape(0);
        std::vector<uint64_t> shapeDims;
        for (int64_t i = 0; i < inputPS.rank().get_length(); i++)
        {
            if (inputPS[i].is_static())
            {
                shapeDims.push_back(inputPS[i].get_length());
            }
            else
            {
                shapeDims.push_back(std::numeric_limits<uint64_t>::max());
            }
        }

        return {MaxValue(shapeDims, 0)};
    }

    vector<MaxValue> exec_gather(Node* node, vector<MaxValue>& inputs)
    {
        auto gather = as_type<op::v1::Gather>(node);

        const auto& indices =
            as_type_ptr<op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
        const auto& axis =
            as_type_ptr<op::v0::Constant>(node->input_value(2).get_node_shared_ptr());

        if (!indices || !axis)
        {
            return {MaxValue()};
        }

        if (gather->get_axis() != 0)
        {
            return {MaxValue()};
        }

        const auto& indicesVec = indices->cast_vector<int64_t>();
        if (indicesVec.size() != 1 ||
            indicesVec[0] >= static_cast<int64_t>(inputs[0].m_slices.size()))
        {
            return {MaxValue()};
        }

        return {MaxValue(inputs[0].m_slices[indicesVec[0]])};
    }

    vector<MaxValue> exec_nop(Node* node, vector<MaxValue>& inputs) { return {inputs.at(0)}; }
} // namespace

pair<bool, uint64_t> ngraph::maximum_value(const Output<Node>& value)
{
    static Evaluator<MaxValue>::op_handler_map handlers = {
        {op::v0::Concat::type_info, exec_concat},
        {op::v0::Constant::type_info, exec_constant},
        {op::v0::Convert::type_info, exec_nop},
        {op::v1::Gather::type_info, exec_gather},
        {op::v1::Minimum::type_info, exec_minimum},
        {op::v1::ReduceMin::type_info, exec_reduce_min},
        {op::v1::Reshape::type_info, exec_nop},
        {op::v3::ShapeOf::type_info, exec_shape_of},
        {op::v0::Squeeze::type_info, exec_nop},
        {op::v0::Unsqueeze::type_info, exec_nop}};
    Evaluator<MaxValue>::value_map value_map;
    Evaluator<MaxValue> evaluator(handlers, value_map);
    auto val = evaluator.evaluate(value);
    return pair<bool, uint64_t>(val.m_value < numeric_limits<uint64_t>::max(), val.m_value);
}

void ngraph::evaluate_nodes(std::map<RawNodeOutput, HostTensorPtr>& value_map,
                            std::map<RawNodeOutput, HostTensorPtr>& output_tensor_map,
                            const OutputVector& outputs,
                            const EvaluationContext& evaluation_context)
{
    Evaluator<HostTensorPtr> evaluator({}, value_map);
    evaluator.set_univeral_handler(
        [&output_tensor_map, &evaluation_context](
            Node* node, const HostTensorVector& input_tensors) -> HostTensorVector {
            HostTensorVector output_tensors;
            for (const auto& v : node->outputs())
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
            if (node->evaluate(output_tensors, input_tensors, evaluation_context))
            {
                return output_tensors;
            }
            else
            {
                NGRAPH_CHECK(false, "Evaluation failed on ", node);
            }
        });
    for (const auto& value : outputs)
    {
        evaluator.evaluate(value);
    }
}

bool could_propagate(const Output<Node>& output, std::vector<Node*>& order)
{
    bool status = true;

    std::deque<Node*> nodes_to_calculate = {output.get_node()};
    order.push_back(output.get_node());

    while (status && !nodes_to_calculate.empty())
    {
        auto current_node = nodes_to_calculate.front();
        nodes_to_calculate.pop_front();

        if (current_node->inputs().empty() && !is_type<op::Constant>(current_node))
            status = false;
        else if (!is_type<op::v0::ShapeOf>(current_node) && !is_type<op::v3::ShapeOf>(current_node))
        {
            // not a leaf, not a shape_of -- continue to search
            for (const auto& input_value : current_node->input_values())
            {
                const auto& input_node = input_value.get_node();
                order.push_back(input_node);
                nodes_to_calculate.push_front(input_node);
            }
        }
    }
    return status;
}

void propagate_rt_info(Node* node, const Output<Node>& final_port)
{
    auto node_outputs = node->outputs();
    bool same_outputs =
        std::all_of(node_outputs.begin(), node_outputs.end(), [](const Output<Node>& output) {
            return output.get_tensor().has_and_set_bound();
        });
    if (same_outputs && op::is_constant(node)) // constant should not propagate it's rt_info
    {
        std::unordered_set<Node*> stop_nodes;
        for (const auto& in : final_port.get_target_inputs())
            stop_nodes.insert(in.get_node());

        auto curr_node = node->shared_from_this();
        for (const auto& output : node_outputs)
        {
            if (output == final_port)
                continue;
            for (auto& in : output.get_target_inputs())
            {
                if (stop_nodes.count(in.get_node()))
                    continue;
                auto consumer = in.get_node()->shared_from_this();
                // FIXME: Here we have a WA in order to save some original fields
                // if we have conflicts because Variant merge doesn't work.
                // We can restore original fields because we don't change the operation
                auto orig_rt_info = consumer->get_rt_info();

                copy_runtime_info({curr_node, consumer}, consumer);

                auto& rt_info = consumer->get_rt_info();
                for (const auto& it : orig_rt_info)
                {
                    if (rt_info.find(it.first) == rt_info.end())
                        rt_info[it.first] = it.second;
                }
            }
        }
    }
}

HostTensorPtr evaluate_bound(const Output<Node>& output, bool is_upper)
{
    // bound is already set in the tensor
    if (is_upper && output.get_tensor().get_upper_value() != nullptr)
        return output.get_tensor().get_upper_value();
    if (!is_upper && output.get_tensor().get_lower_value() != nullptr)
        return output.get_tensor().get_lower_value();

    std::vector<Node*> order;
    if (could_propagate(output, order))
    {
        reverse(order.begin(), order.end());
        for (const auto& node : order)
        {
            HostTensorVector outputs;
            for (const auto& out : node->outputs())
                outputs.push_back(std::make_shared<HostTensor>(out));
            if (is_upper ? node->evaluate_upper(outputs) : node->evaluate_lower(outputs))
            {
                const auto& input_values = node->input_values();
                bool same_inputs = std::all_of(
                    input_values.begin(), input_values.end(), [](const Output<Node>& input) {
                        return input.get_tensor().has_and_set_bound();
                    });
                for (size_t i = 0; i < outputs.size(); ++i)
                {
                    // TODO: should we skip setting value for tensors that have only one consumer?
                    if ((same_inputs || is_upper) &&
                        node->get_output_tensor(i).get_upper_value() == nullptr)
                        node->get_output_tensor(i).set_upper_value(outputs[i]);
                    if ((same_inputs || !is_upper) &&
                        node->get_output_tensor(i).get_lower_value() == nullptr)
                        node->get_output_tensor(i).set_lower_value(outputs[i]);
                }
                for (const auto& input : input_values)
                    if (input.get_target_inputs().size() == 1)
                        input.get_tensor().invalidate_values();
                propagate_rt_info(node, output);
            }
            else
            {
                break;
            }
        }
    }
    if (is_upper)
        return output.get_tensor().get_upper_value();
    else
        return output.get_tensor().get_lower_value();
}

HostTensorPtr ngraph::evaluate_lower_bound(const Output<Node>& output)
{
    return evaluate_bound(output, false);
}

HostTensorPtr ngraph::evaluate_upper_bound(const Output<Node>& output)
{
    return evaluate_bound(output, true);
}

pair<HostTensorPtr, HostTensorPtr> ngraph::evaluate_both_bounds(const Output<Node>& output)
{
    return {evaluate_lower_bound(output), evaluate_upper_bound(output)};
}

bool ngraph::evaluate_as_partial_shape(const Output<Node>& output, PartialShape& pshape)
{
    HostTensorPtr lb, ub;
    std::tie(lb, ub) = evaluate_both_bounds(output);
    bool shape_defined = false;
    if (lb && ub)
    {
        const auto lower_bound = std::make_shared<op::Constant>(lb)->cast_vector<int64_t>();
        const auto upper_bound = std::make_shared<op::Constant>(ub)->cast_vector<int64_t>();
        NGRAPH_CHECK(lower_bound.size() == upper_bound.size());
        vector<Dimension> resulting_pshape(lower_bound.size());
        for (size_t i = 0; i < lower_bound.size(); ++i)
        {
            NGRAPH_CHECK(lower_bound[i] >= 0 && upper_bound[i] >= 0);
            resulting_pshape[i] = {lower_bound[i], upper_bound[i]};
        }
        pshape = PartialShape(resulting_pshape);
        shape_defined = true;
    }
    return shape_defined;
}

bool default_bound_evaluator(const Node* node, const HostTensorVector& output_values, bool is_upper)
{
    HostTensorVector input_tensors;
    for (const auto& input : node->input_values())
    {
        if (auto bound = is_upper ? input.get_tensor().get_upper_value()
                                  : input.get_tensor().get_lower_value())
            input_tensors.push_back(bound);
        else
            return false;
    }
    return node->evaluate(output_values, input_tensors);
}

bool ngraph::default_lower_bound_evaluator(const Node* node, const HostTensorVector& output_values)
{
    return default_bound_evaluator(node, output_values, false);
}

bool ngraph::default_upper_bound_evaluator(const Node* node, const HostTensorVector& output_values)
{
    return default_bound_evaluator(node, output_values, true);
}

shared_ptr<op::Constant> ngraph::get_constant_max_of_type(element::Type_t t)
{
#define NGRAPH_TYPE_TO_MAX_CONST(t)                                                                \
    case t:                                                                                        \
        return op::Constant::create(                                                               \
            t, {}, {std::numeric_limits<typename element_type_traits<t>::value_type>::max()});     \
        break

    switch (t)
    {
        NGRAPH_TYPE_TO_MAX_CONST(element::boolean);
        NGRAPH_TYPE_TO_MAX_CONST(element::bf16);
        NGRAPH_TYPE_TO_MAX_CONST(element::f16);
        NGRAPH_TYPE_TO_MAX_CONST(element::f32);
        NGRAPH_TYPE_TO_MAX_CONST(element::f64);
        NGRAPH_TYPE_TO_MAX_CONST(element::i8);
        NGRAPH_TYPE_TO_MAX_CONST(element::i16);
        NGRAPH_TYPE_TO_MAX_CONST(element::i32);
        NGRAPH_TYPE_TO_MAX_CONST(element::i64);
        NGRAPH_TYPE_TO_MAX_CONST(element::u1);
        NGRAPH_TYPE_TO_MAX_CONST(element::u8);
        NGRAPH_TYPE_TO_MAX_CONST(element::u16);
        NGRAPH_TYPE_TO_MAX_CONST(element::u32);
        NGRAPH_TYPE_TO_MAX_CONST(element::u64);

    case element::undefined:
    case element::dynamic:
    default: return nullptr;
    }
}

shared_ptr<op::Constant> ngraph::get_constant_min_of_type(element::Type_t t)
{
#define NGRAPH_TYPE_TO_MIN_CONST(t)                                                                \
    case t:                                                                                        \
        return op::Constant::create(                                                               \
            t, {}, {std::numeric_limits<typename element_type_traits<t>::value_type>::min()});     \
        break

    switch (t)
    {
        NGRAPH_TYPE_TO_MIN_CONST(element::boolean);
        NGRAPH_TYPE_TO_MIN_CONST(element::bf16);
        NGRAPH_TYPE_TO_MIN_CONST(element::f16);
        NGRAPH_TYPE_TO_MIN_CONST(element::f32);
        NGRAPH_TYPE_TO_MIN_CONST(element::f64);
        NGRAPH_TYPE_TO_MIN_CONST(element::i8);
        NGRAPH_TYPE_TO_MIN_CONST(element::i16);
        NGRAPH_TYPE_TO_MIN_CONST(element::i32);
        NGRAPH_TYPE_TO_MIN_CONST(element::i64);
        NGRAPH_TYPE_TO_MIN_CONST(element::u1);
        NGRAPH_TYPE_TO_MIN_CONST(element::u8);
        NGRAPH_TYPE_TO_MIN_CONST(element::u16);
        NGRAPH_TYPE_TO_MIN_CONST(element::u32);
        NGRAPH_TYPE_TO_MIN_CONST(element::u64);

    case element::undefined:
    case element::dynamic:
    default: return nullptr;
    }
}

HostTensorPtr equality_mask(const HostTensorPtr& tensor, const shared_ptr<op::Constant>& constant)
{
    auto mask = std::make_shared<HostTensor>(element::boolean, tensor->get_shape());
    const auto& param =
        std::make_shared<op::Parameter>(tensor->get_element_type(), tensor->get_shape());
    op::v1::Equal(param, constant, ngraph::op::AutoBroadcastSpec::NUMPY)
        .evaluate({mask}, {tensor, std::make_shared<HostTensor>(constant)});
    return mask;
}

HostTensorPtr or_tensor(const HostTensorPtr& lhs, const HostTensorPtr& rhs)
{
    auto result = std::make_shared<HostTensor>();
    op::v1::LogicalOr(std::make_shared<op::Parameter>(lhs->get_element_type(), lhs->get_shape()),
                      std::make_shared<op::Parameter>(rhs->get_element_type(), rhs->get_shape()),
                      ngraph::op::AutoBroadcastSpec::NUMPY)
        .evaluate({result}, {lhs, rhs});
    return result;
}

bool ngraph::interval_bound_evaluator(const Node* node,
                                      const HostTensorVector& lower_output_values,
                                      const HostTensorVector& upper_output_values)
{
    // TODO: relax for n inputs ?
    NGRAPH_CHECK(lower_output_values.size() == upper_output_values.size());
    NGRAPH_CHECK(node->get_input_size() == 2);

    const auto num_of_outputs = node->get_output_size();
    std::shared_ptr<HostTensor> low_0 = evaluate_lower_bound(node->get_input_source_output(0));
    std::shared_ptr<HostTensor> low_1 = evaluate_lower_bound(node->get_input_source_output(1));
    std::shared_ptr<HostTensor> up_0 = evaluate_upper_bound(node->get_input_source_output(0));
    std::shared_ptr<HostTensor> up_1 = evaluate_upper_bound(node->get_input_source_output(1));
    std::set<HostTensorVector> input_variants = {
        {low_0, low_1}, {low_0, up_1}, {up_0, low_1}, {up_0, up_1}};

    for (const auto& variant_of_input_vector : input_variants)
        for (const auto& input_tensor : variant_of_input_vector)
            if (input_tensor == nullptr)
                return false;

    if (input_variants.size() == 1)
        return node->evaluate(upper_output_values, *input_variants.begin()) &&
               node->evaluate(lower_output_values, *input_variants.begin());

    auto zero = op::v0::Constant::create(element::i64, {1}, {0});
    std::vector<HostTensorVector> unsqueezed_output_variants;
    for (auto& input_variant : input_variants)
    {
        HostTensorVector vector_of_output_variants;
        for (const auto& output : lower_output_values)
            vector_of_output_variants.push_back(std::make_shared<HostTensor>(
                output->get_element_type(), output->get_partial_shape()));

        node->evaluate(vector_of_output_variants, input_variant);

        HostTensorVector vector_of_unsqueezed_output_variants;
        for (const auto& output : vector_of_output_variants)
        {
            if (!output)
                return false;
            auto unsqueezed_shape = output->get_shape();
            unsqueezed_shape.insert(unsqueezed_shape.begin(), 1);
            const auto unsqueezed =
                make_shared<HostTensor>(output->get_element_type(), unsqueezed_shape);
            op::v0::Unsqueeze().evaluate({unsqueezed}, {output, make_shared<HostTensor>(zero)});
            vector_of_unsqueezed_output_variants.push_back(unsqueezed);
        }
        unsqueezed_output_variants.push_back(vector_of_unsqueezed_output_variants);
    }

    auto input_0_maximum_value = get_constant_max_of_type(low_0->get_element_type());
    auto input_1_maximum_value = get_constant_max_of_type(low_1->get_element_type());
    if (input_0_maximum_value == nullptr || input_1_maximum_value == nullptr)
        return false;

    auto input_0_low_dyn_mask = equality_mask(low_0, input_0_maximum_value);
    auto input_0_up_dyn_mask = equality_mask(up_0, input_0_maximum_value);
    auto input_1_low_dyn_mask = equality_mask(low_1, input_1_maximum_value);
    auto input_1_up_dyn_mask = equality_mask(up_1, input_1_maximum_value);

    auto final_input_dyn_mask = or_tensor(or_tensor(input_0_low_dyn_mask, input_0_up_dyn_mask),
                                          or_tensor(input_1_low_dyn_mask, input_1_up_dyn_mask));

    bool fully_defined = true;
    for (size_t i = 0; i < num_of_outputs; ++i)
    {
        HostTensorVector all_variants_for_ith_output;
        for (const auto& unsqueezed_output_variant : unsqueezed_output_variants)
            all_variants_for_ith_output.push_back(unsqueezed_output_variant[i]);

        auto concated_shape = all_variants_for_ith_output[0]->get_shape();
        concated_shape[0] = all_variants_for_ith_output.size();
        auto concated = make_shared<HostTensor>(all_variants_for_ith_output[0]->get_element_type(),
                                                concated_shape);
        auto concat = op::Concat();
        concat.set_axis(0);
        concat.evaluate({concated}, all_variants_for_ith_output);

        auto fake_param = make_shared<op::Parameter>(
            all_variants_for_ith_output[0]->get_element_type(), concated_shape);
        auto reduce_min_op = op::v1::ReduceMin(fake_param, zero, false);
        reduce_min_op.evaluate({lower_output_values[i]}, {concated, make_shared<HostTensor>(zero)});
        auto reduce_max_op = op::v1::ReduceMax(fake_param, zero, false);
        reduce_max_op.evaluate({upper_output_values[i]}, {concated, make_shared<HostTensor>(zero)});

        if (upper_output_values[i] == nullptr)
            fully_defined = false;
        else
        {
            auto output_maximum_value =
                get_constant_max_of_type(upper_output_values[i]->get_element_type());
            op::v1::Select().evaluate({upper_output_values[i]},
                                      {final_input_dyn_mask,
                                       std::make_shared<HostTensor>(output_maximum_value),
                                       upper_output_values[i]});
            node->get_output_tensor(i).set_upper_value(upper_output_values[i]);
        }
        if (lower_output_values[i] == nullptr)
            fully_defined = false;
        else
        {
            auto output_minimum_value =
                op::Constant::create(lower_output_values[i]->get_element_type(), {}, {0});
            // Can not set to get_constant_min_of_type(lower_output_values[i]->get_element_type())
            // yet
            op::v1::Select().evaluate({lower_output_values[i]},
                                      {final_input_dyn_mask,
                                       std::make_shared<HostTensor>(output_minimum_value),
                                       lower_output_values[i]});
            node->get_output_tensor(i).set_lower_value(lower_output_values[i]);
        }
    }
    return fully_defined;
}

bool ngraph::host_tensor_is_positive(const HostTensorPtr& bound)
{
    const auto bound_constant = std::make_shared<op::Constant>(bound);
    const auto zero_constant = op::Constant::create(bound->get_element_type(), {1}, {0});
    OutputVector greater(1);
    bool folded = std::make_shared<op::v1::Greater>(bound_constant, zero_constant)
                      ->constant_fold(greater, {bound_constant, zero_constant});
    NGRAPH_CHECK(folded);

    auto axes_vector = std::vector<int64_t>(greater[0].get_shape().size());
    std::iota(axes_vector.begin(), axes_vector.end(), 0);
    const auto axes = op::Constant::create(element::i64, {axes_vector.size()}, axes_vector);
    OutputVector all(1);
    folded = std::make_shared<op::v1::ReduceLogicalAnd>(greater[0], axes)
                 ->constant_fold(all, {greater[0], axes});
    NGRAPH_CHECK(folded && is_type<op::Constant>(all[0].get_node_shared_ptr()));
    const auto result =
        std::dynamic_pointer_cast<op::Constant>(all[0].get_node_shared_ptr())->cast_vector<bool>();
    NGRAPH_CHECK(all[0].get_shape() == Shape{});
    return result[0];
}

bool ngraph::has_and_set_equal_bounds(const Output<Node>& source)
{
    if (op::is_constant(source.get_node_shared_ptr()))
        return true;
    HostTensorPtr lb, ub;
    std::tie(lb, ub) = evaluate_both_bounds(source);
    return lb && lb == ub;
}

shared_ptr<op::Constant> ngraph::get_constant_from_source(const Output<Node>& source)
{
    if (!has_and_set_equal_bounds(source))
        return nullptr;
    if (const auto& c = as_type_ptr<op::Constant>(source.get_node_shared_ptr()))
        return c;
    return std::make_shared<op::Constant>(source.get_tensor().get_upper_value());
}

bool ngraph::validate_host_tensor_vector(const HostTensorVector& tensor_vector, const size_t& size)
{
    if (tensor_vector.size() != size)
        return false;
    return std::all_of(tensor_vector.begin(), tensor_vector.end(), [](const HostTensorPtr& t) {
        return t != nullptr;
    });
}
