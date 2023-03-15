// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_conv.hpp"

#include "bound_evaluate.hpp"
#include "group_convolution_backprop_shape_inference.hpp"
#include "group_convolution_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;

//------------------------------------------------------------------------------
//                        v1::GroupConvolution
//------------------------------------------------------------------------------
namespace ov {
op::v1::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
                                           const Output<Node>& filters,
                                           const Strides& strides,
                                           const CoordinateDiff& pads_begin,
                                           const CoordinateDiff& pads_end,
                                           const Strides& dilations,
                                           const PadType& auto_pad)
    : Op({data_batch, filters}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad) {
    constructor_validate_and_infer_types();
}

bool op::v1::GroupConvolution::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_GroupConvolution_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void op::v1::GroupConvolution::validate_and_infer_types() {
    OV_OP_SCOPE(v1_GroupConvolution_validate_and_infer_types);
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
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    m_num_spatial = ov::util::dim::inf_bound;
    const auto input_shapes = get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, result_et, output_shapes[0]);
    if (input_shapes[0].rank().is_static() && input_shapes[1].rank().is_static()) {
        m_num_spatial = convolution::get_num_spatial(this, input_shapes);
    }
}

shared_ptr<Node> op::v1::GroupConvolution::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_GroupConvolution_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::GroupConvolution>(new_args.at(0),
                                             new_args.at(1),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_dilations,
                                             m_auto_pad);
}

//------------------------------------------------------------------------------
//                        v1::GroupConvolutionBackpropData
//------------------------------------------------------------------------------

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData()
    : Op(),
      m_strides(),
      m_dilations(),
      m_pads_begin(),
      m_pads_end(),
      m_auto_pad(),
      m_output_padding() {}

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(const Output<Node>& data,
                                                                   const Output<Node>& filters,
                                                                   const Output<Node>& output_shape,
                                                                   const Strides& strides,
                                                                   const CoordinateDiff& pads_begin,
                                                                   const CoordinateDiff& pads_end,
                                                                   const Strides& dilations,
                                                                   const PadType& auto_pad,
                                                                   const CoordinateDiff& output_padding)
    : Op({data, filters, output_shape}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad),
      m_output_padding(output_padding) {
    ov::mark_as_precision_sensitive(input(2));
    constructor_validate_and_infer_types();
}

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(const Output<Node>& data,
                                                                   const Output<Node>& filters,
                                                                   const Output<Node>& output_shape,
                                                                   const Strides& strides,
                                                                   const Strides& dilations,
                                                                   const PadType& auto_pad,
                                                                   const CoordinateDiff& output_padding)
    : GroupConvolutionBackpropData(data,
                                   filters,
                                   output_shape,
                                   strides,
                                   CoordinateDiff(),
                                   CoordinateDiff(),
                                   dilations,
                                   auto_pad,
                                   output_padding) {}

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(const Output<Node>& data,
                                                                   const Output<Node>& filters,
                                                                   const Strides& strides,
                                                                   const CoordinateDiff& pads_begin,
                                                                   const CoordinateDiff& pads_end,
                                                                   const Strides& dilations,
                                                                   const PadType& auto_pad,
                                                                   const CoordinateDiff& output_padding)
    : Op({data, filters}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad),
      m_output_padding(output_padding) {
    constructor_validate_and_infer_types();
}

bool op::v1::GroupConvolutionBackpropData::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_GroupConvolutionBackpropData_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("output_padding", m_output_padding);
    return true;
}

bool op::v1::GroupConvolutionBackpropData::is_dynamic() const {
    return Node::is_dynamic() || (get_input_size() == 3 && !has_and_set_equal_bounds(input_value(2)));
}

const ov::PartialShape op::v1::GroupConvolutionBackpropData::get_convolution_output_shape() const {
    auto shape = PartialShape::dynamic();

    if (get_input_size() != 3 || !evaluate_as_partial_shape(input_value(2), shape)) {
        const auto& data_pshape = get_input_partial_shape(0);
        const auto& filter_pshape = get_input_partial_shape(1);

        if (data_pshape.rank().is_static()) {
            shape.resize(data_pshape.rank().get_length() - convolution::spatial_dim_offset);
        } else if (filter_pshape.rank().is_static()) {
            shape.resize(filter_pshape.rank().get_length() - convolution::spatial_dim_offset);
        }
    }

    return shape;
}

void op::v1::GroupConvolutionBackpropData::set_output_shape(const ov::Shape& shape) {
    this->input(2).replace_source_output(
        op::v0::Constant::create(this->get_input_element_type(2), ov::Shape{shape.size()}, shape)->output(0));
}

void op::v1::GroupConvolutionBackpropData::infer_conv_backprop_output_spatial_shape(
    const vector<Dimension>& input_data_shape,
    const vector<Dimension>& filters_shape,
    const Strides& strides,
    const Strides& dilations,
    const CoordinateDiff& pads_begin,
    const CoordinateDiff& pads_end,
    const CoordinateDiff& output_padding,
    vector<Dimension>& output_spatial_shape) {
    size_t num_spatial_dims = input_data_shape.size();
    OPENVINO_ASSERT(filters_shape.size() == num_spatial_dims && strides.size() == num_spatial_dims &&
                    dilations.size() == num_spatial_dims && pads_begin.size() == num_spatial_dims &&
                    pads_end.size() == num_spatial_dims && output_padding.size() == num_spatial_dims);

    for (size_t i = 0; i < num_spatial_dims; ++i) {
        if (input_data_shape[i].is_static() && filters_shape[i].is_static()) {
            int64_t val = strides[i] * (input_data_shape[i].get_length() - 1) +
                          dilations[i] * (filters_shape[i].get_length() - 1) + 1 - pads_begin[i] - pads_end[i] +
                          output_padding[i];
            output_spatial_shape.emplace_back(val);
        } else {
            output_spatial_shape.push_back(Dimension::dynamic());
        }
    }
}

void op::v1::GroupConvolutionBackpropData::validate_and_infer_types() {
    OV_OP_SCOPE(v1_GroupConvolutionBackpropData_validate_and_infer_types);
    const auto& data_et = get_input_element_type(0);
    const auto& filters_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_et, filters_et),
                          "Element types for data batch and filters do not match (data batch element type: ",
                          data_et,
                          ", filters element type: ",
                          filters_et,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    bool output_shape_input_present = get_input_size() == 3;
    if (output_shape_input_present) {
        const element::Type output_shape_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              output_shape_et.is_integral_number(),
                              "Element type for output shape should be of integer type ",
                              "(output_shape element type: ",
                              output_shape_et,
                              ").");
    }

    m_num_spatial = ov::util::dim::inf_bound;
    const auto input_shapes = get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, result_et, output_shapes[0]);
    if (input_shapes[0].rank().is_static() && input_shapes[1].rank().is_static()) {
        m_num_spatial = convolution::get_num_spatial(this, input_shapes);
    }

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
}

shared_ptr<Node> op::v1::GroupConvolutionBackpropData::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_GroupConvolutionBackpropData_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return make_shared<v1::GroupConvolutionBackpropData>(new_args.at(0),
                                                             new_args.at(1),
                                                             new_args.at(2),
                                                             m_strides,
                                                             m_pads_begin,
                                                             m_pads_end,
                                                             m_dilations,
                                                             m_auto_pad,
                                                             m_output_padding);
    } else {
        return make_shared<v1::GroupConvolutionBackpropData>(new_args.at(0),
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
