// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include "bound_evaluate.hpp"
#include "convolution_backprop_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;

namespace ov {
// *** Convolution OP SET 1 ***
op::v1::Convolution::Convolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const PadType& auto_pad)
    : ConvolutionFwdPropBase({data_batch, filters}, strides, pads_begin, pads_end, dilations, auto_pad) {
    constructor_validate_and_infer_types();
}

bool op::v1::Convolution::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Convolution_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::v1::Convolution::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Convolution_validate_and_infer_types);
    const auto& data_batch_et = get_input_element_type(0);
    const auto& filters_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_batch_et, filters_et),
                          "Element types for data batch and filters do not match (data batch element type: ",
                          data_batch_et,
                          ", filters element type: ",
                          filters_et,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element types must be numeric. Got: ",
                          result_et);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto num_spatial = convolution::calculate_num_spatial(this, input_shapes);
    if (num_spatial != util::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }

    const auto output_shapes = shape_infer(this, input_shapes, m_pads_begin, m_pads_end);
    set_output_type(0, result_et, output_shapes[0]);
    set_num_spatial(num_spatial, input_shapes);
}

shared_ptr<Node> op::v1::Convolution::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Convolution_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Convolution>(new_args.at(0),
                                        new_args.at(1),
                                        m_strides,
                                        m_pads_begin,
                                        m_pads_end,
                                        m_dilations,
                                        m_auto_pad);
}

// *** ConvolutionBackpropData OP SET 1 ***
op::v1::ConvolutionBackpropData::ConvolutionBackpropData(const Output<Node>& data,
                                                         const Output<Node>& filters,
                                                         const Output<Node>& output_shape,
                                                         const Strides& strides,
                                                         const CoordinateDiff& pads_begin,
                                                         const CoordinateDiff& pads_end,
                                                         const Strides& dilations,
                                                         const PadType& auto_pad,
                                                         const CoordinateDiff& output_padding)
    : ConvolutionBackPropBase({data, filters, output_shape},
                              strides,
                              pads_begin,
                              pads_end,
                              dilations,
                              auto_pad,
                              output_padding) {
    ov::mark_as_precision_sensitive(input(2));
    constructor_validate_and_infer_types();
}

bool op::v1::ConvolutionBackpropData::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_ConvolutionBackpropData_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("output_padding", m_output_padding);
    return true;
}

op::v1::ConvolutionBackpropData::ConvolutionBackpropData(const Output<Node>& data,
                                                         const Output<Node>& filters,
                                                         const Strides& strides,
                                                         const CoordinateDiff& pads_begin,
                                                         const CoordinateDiff& pads_end,
                                                         const Strides& dilations,
                                                         const PadType& auto_pad,
                                                         const CoordinateDiff& output_padding)
    : ConvolutionBackPropBase({data, filters}, strides, pads_begin, pads_end, dilations, auto_pad, output_padding) {
    constructor_validate_and_infer_types();
}

bool op::v1::ConvolutionBackpropData::is_dynamic() const {
    return Node::is_dynamic() || (get_input_size() == 3 && !has_and_set_equal_bounds(input_value(2)));
}

const ov::PartialShape op::v1::ConvolutionBackpropData::get_output_shape() const {
    auto shape = PartialShape::dynamic();

    if (get_input_size() < 3 || !ov::util::evaluate_as_partial_shape(input_value(2), shape)) {
        const auto& data_rank = get_input_partial_shape(0).rank();
        const auto& filter_rank = get_input_partial_shape(1).rank();

        if (data_rank.is_static()) {
            shape.resize(data_rank.get_length() - convolution::spatial_dim_offset);
        } else if (filter_rank.is_static()) {
            shape.resize(filter_rank.get_length() - convolution::spatial_dim_offset);
        } else if (get_input_size() == 3) {
            const auto& out_spatial_shape = get_input_partial_shape(2);
            if (out_spatial_shape.is_static()) {
                shape.resize(out_spatial_shape[0].get_length());
            }
        }
    }

    return shape;
}

void op::v1::ConvolutionBackpropData::set_output_shape(const ov::Shape& shape) {
    element::Type_t et = (get_input_size() == 3) ? get_input_element_type(2) : element::i64;
    if (get_input_size() == 0) {
        // Add dummy inputs when adding output shape and op has no inputs at all.
        auto dummy = std::make_shared<v0::Constant>(et, ov::Shape{0});
        set_argument(0, dummy);
        set_argument(1, dummy);
    }
    set_argument(2, v0::Constant::create(et, Shape{shape.size()}, shape));
}

void op::v1::ConvolutionBackpropData::validate_and_infer_types() {
    OV_OP_SCOPE(v1_ConvolutionBackpropData_validate_and_infer_types);
    const auto& delta_et = get_input_element_type(0);
    const auto& filters_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, delta_et, filters_et),
                          "Element types for data batch and filters do not match (data batch element type: ",
                          delta_et,
                          ", filters element type: ",
                          filters_et,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    if (get_input_size() == 3) {
        const auto& output_shape_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              output_shape_et.is_integral_number(),
                              "Element type for output shape should be of integer type ",
                              "(output_shape element type: ",
                              output_shape_et,
                              ").");
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto out_spatial_shape = get_output_shape();
    auto num_spatial = convolution::calculate_num_spatial(this, input_shapes, out_spatial_shape);

    if (num_spatial != util::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }

    const auto output_shapes = shape_infer(this, input_shapes, m_pads_begin, m_pads_end);
    set_output_type(0, result_et, output_shapes[0]);
    set_num_spatial(num_spatial, input_shapes);

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
}

shared_ptr<Node> op::v1::ConvolutionBackpropData::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ConvolutionBackpropData_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return make_shared<v1::ConvolutionBackpropData>(new_args.at(0),
                                                        new_args.at(1),
                                                        new_args.at(2),
                                                        m_strides,
                                                        m_pads_begin,
                                                        m_pads_end,
                                                        m_dilations,
                                                        m_auto_pad,
                                                        m_output_padding);
    } else {
        return make_shared<v1::ConvolutionBackpropData>(new_args.at(0),
                                                        new_args.at(1),
                                                        m_strides,
                                                        m_pads_begin,
                                                        m_pads_end,
                                                        m_dilations,
                                                        m_auto_pad,
                                                        m_output_padding);
    }
}
}  // namespace ov
