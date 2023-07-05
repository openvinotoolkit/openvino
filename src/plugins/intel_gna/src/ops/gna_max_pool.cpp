// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_max_pool.hpp"

#include <assert.h>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"

namespace ov {
namespace intel_gna {
namespace op {
//
// Infers the output batch shape and element type for batched pooling fprop.
//
ov::PartialShape infer_batched_pooling_forward(const ov::Node* node,
                                               const ov::PartialShape& data_batch_shape,
                                               const ov::CoordinateDiff& data_padding_below,
                                               const ov::CoordinateDiff& data_padding_above,
                                               const ov::PartialShape& window_shape,
                                               const ov::Strides& window_strides,
                                               bool is_window_all_in_padding_allowed,
                                               bool ceil_mode,
                                               const ov::Strides& window_dilation);

//
// Infers the output batch shape and element type for batched pooling fprop.
//
ov::PartialShape infer_batched_pooling_forward(const ov::Node* node,
                                               const ov::PartialShape& data_batch_shape,
                                               const ov::CoordinateDiff& data_padding_below,
                                               const ov::CoordinateDiff& data_padding_above,
                                               const ov::PartialShape& window_shape,
                                               const ov::Strides& window_strides,
                                               bool is_window_all_in_padding_allowed,
                                               bool ceil_mode,
                                               const ov::Strides& window_dilation) {
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

    ov::Dimension batch_size{ov::Dimension::dynamic()};
    ov::Dimension channel_count{ov::Dimension::dynamic()};
    ov::PartialShape data_output_spatial_shape{ov::PartialShape::dynamic(data_spatial_shape.rank())};

    if (data_batch_shape.rank().is_static()) {
        batch_size = data_batch_shape[0];
        channel_count = *(data_batch_shape.end() - 1);  // EMUTEX fix NCHW -> NHWC from data_batch_shape[1]

        for (int64_t i = 0; i < data_spatial_shape.rank().get_length(); i++) {
            data_spatial_shape[i] =
                data_batch_shape[i +
                                 1];  // EMUTEX fix NCHW -> NHWC from data_spatial_shape[i] = data_batch_shape[i + 2]
        }

        NODE_VALIDATION_CHECK(node, batch_size.is_dynamic() || batch_size.get_length() > 0, "Batch size is zero.");

        NODE_VALIDATION_CHECK(node,
                              channel_count.is_dynamic() || channel_count.get_length() > 0,
                              "Channel count is zero.");

        // For pooling ops we don't need dilation, so we fill in the identity value (all 1).
        ov::Strides data_dilation(data_spatial_shape.rank().get_length(), 1);
        ov::Strides dilations = window_dilation;
        // if the window_dilation was not specified, generate the default value (no dilations)
        if (window_dilation.empty()) {
            // dilations equal to 1 for each spatial axis mean that the window is not dilated
            dilations = ov::Strides(data_spatial_shape.rank().get_length(), 1);
        }

        data_output_spatial_shape = ngraph::infer_windowed_reduction_output_shape(node,
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
    *(data_batch_output_shape.end() - 1) =
        channel_count;  // EMUTEX fix NCHW -> NHWC data_batch_output_shape[1] = channel_count;

    for (int64_t i = 0; i < data_spatial_shape.rank().get_length(); i++) {
        data_batch_output_shape[i + 1] =
            data_output_spatial_shape[i];  // EMUTEX fix NCHW -> NHWC data_batch_output_shape[i + 2] =
                                           // data_output_spatial_shape[i];
    }

    return data_batch_output_shape;
}

GNAMaxPool::GNAMaxPool(const ov::Output<ov::Node>& arg,
                       const ov::Strides& strides,
                       const ov::Shape& pads_begin,
                       const ov::Shape& pads_end,
                       const ov::Shape& kernel,
                       const ov::op::RoundingType rounding_type,
                       const ov::op::PadType auto_pad)
    : Op({arg}),
      m_kernel(kernel),
      m_strides(strides),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad),
      m_rounding_type(rounding_type) {
    constructor_validate_and_infer_types();
}

bool GNAMaxPool::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("rounding_type", m_rounding_type);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void GNAMaxPool::validate_and_infer_types() {
    if (0 == m_strides.size()) {
        m_strides = ov::Strides(m_kernel.size(), 1);
    }

    if (0 == m_pads_begin.size()) {
        m_pads_begin = ov::Shape(m_kernel.size(), 0);
    }

    if (0 == m_pads_end.size()) {
        m_pads_end = ov::Shape(m_kernel.size(), 0);
    }

    const ov::PartialShape& arg_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this,
        arg_shape.rank().compatible(3) || arg_shape.rank().compatible(4) || arg_shape.rank().compatible(5),
        "Expected a 3D, 4D or 5D tensor for the input. Got: ",
        arg_shape);

    if (arg_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_end.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected pads_end size to be equal to input size - 2. Got: ",
                              m_pads_end.size());

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_begin.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected pads_begin size to be equal to input size - 2. Got: ",
                              m_pads_begin.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_kernel.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected kernel size to be equal to input size - 2. Got: ",
                              m_kernel.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_strides.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected strides size to be equal to input size - 2. Got: ",
                              m_strides.size());
    }

    const ov::PartialShape output_shape = infer_output_shape(ov::Strides{});  // no dilations of the filter window

    set_output_type(0, get_input_element_type(0), output_shape);
}

ov::PartialShape GNAMaxPool::infer_output_shape(const ov::Strides& dilations) {
    const auto& arg_shape = get_input_partial_shape(0);

    bool update_auto_padding_succeed = true;

    if (m_auto_pad == ov::op::PadType::SAME_UPPER || m_auto_pad == ov::op::PadType::SAME_LOWER) {
        const auto filter_dilations = dilations.empty() ? ov::Strides(m_kernel.size(), 1) : dilations;
        update_auto_padding_succeed = update_auto_padding(arg_shape, filter_dilations, m_pads_end, m_pads_begin);
    }
    if (m_auto_pad == ov::op::PadType::VALID) {
        m_pads_end = ov::Shape(m_pads_end.size(), 0);
        m_pads_begin = ov::Shape(m_pads_begin.size(), 0);
    }

    auto output_shape = ov::PartialShape::dynamic();
    if (update_auto_padding_succeed) {
        ov::CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
        ov::CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());
        output_shape = ov::intel_gna::op::infer_batched_pooling_forward(this,
                                                                        get_input_partial_shape(0),
                                                                        pads_begin,
                                                                        pads_end,
                                                                        m_kernel,
                                                                        m_strides,
                                                                        true,
                                                                        m_rounding_type == ov::op::RoundingType::CEIL,
                                                                        dilations);
    } else {
        if (arg_shape.rank().is_static()) {
            output_shape = std::vector<ov::Dimension>(arg_shape.rank().get_max_length(), ov::Dimension::dynamic());
            if (arg_shape[0].is_static()) {
                output_shape[0] = arg_shape[0];  // batch size
            }
            if ((arg_shape.end() - 1)->is_static()) {                // EMUTEX FIXED: from [1] to end() - 1 NCHW -> NHWC
                *(output_shape.end() - 1) = *(arg_shape.end() - 1);  // channel size
            }
        }
    }

    return output_shape;
}

bool GNAMaxPool::update_auto_padding(const ov::PartialShape& in_shape,
                                     const ov::Strides& filter_dilations,
                                     ov::Shape& new_pads_end,
                                     ov::Shape& new_pads_begin) const {
    bool update_auto_padding_succeed = true;
    if (m_auto_pad == ov::op::PadType::SAME_UPPER || m_auto_pad == ov::op::PadType::SAME_LOWER) {
        ov::CoordinateDiff pads_end, pads_begin;
        update_auto_padding_succeed = ngraph::try_apply_auto_padding(in_shape,
                                                                     m_kernel,
                                                                     m_strides,
                                                                     filter_dilations,
                                                                     m_auto_pad,
                                                                     pads_end,
                                                                     pads_begin);
        new_pads_end = ov::Shape(pads_end.begin(), pads_end.end());
        new_pads_begin = ov::Shape(pads_begin.begin(), pads_begin.end());
    }
    return update_auto_padding_succeed;
}

std::shared_ptr<ov::Node> GNAMaxPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GNAMaxPool>(new_args.at(0),
                                        m_strides,
                                        m_pads_begin,
                                        m_pads_end,
                                        m_kernel,
                                        m_rounding_type,
                                        m_auto_pad);
}

}  // namespace op
}  // namespace intel_gna
}  // namespace ov