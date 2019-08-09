// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <memory>

#include "group_conv_bias.hpp"

#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static void validate_groupconvbias_shapes(const Shape& input_shape,
                                          const Shape& filters_shape,
                                          const Shape& bias_shape,
                                          const Shape& output_shape,
                                          size_t groups) {
    //  Input - N, C, H, W
    //  Filter - O, I, H, W
    //  Output - N, C, H, W
    const size_t INPUT_C = 1;
    const size_t FILTER_OC = 0;
    const size_t FILTER_IC = 1;
    const size_t OUTPUT_C = 1;

    if (bias_shape.size() != 1) {
        throw ngraph_error("GroupConvolutionBias bias is expected to be 1D, but has shape: " +
                           vector_to_string(bias_shape));
    }

    if (bias_shape[0] != filters_shape[FILTER_OC]) {
        throw ngraph_error(
            "GroupConvolutionBias bias element size does not match number of filters. bias_size "
            "= " +
            std::to_string(bias_shape[0]) + ", num_filters = " + std::to_string(filters_shape[0]));
    }

    if (input_shape[INPUT_C] != groups * filters_shape[FILTER_IC]) {
        throw ngraph_error(
            "Mismatch between GroupConvolutionBias input and filter channels: "
            " data channels=" +
            std::to_string(input_shape[INPUT_C]) + ", filter channels= " +
            std::to_string(filters_shape[FILTER_IC]) + ", groups= " + std::to_string(groups));
    }

    if (output_shape[OUTPUT_C] != filters_shape[FILTER_OC]) {
        throw ngraph_error(
            "Mismatch between GroupConvolutionBias output and filter channels: "
            " data channels=" +
            std::to_string(output_shape[OUTPUT_C]) + ", filter channels= " +
            std::to_string(filters_shape[FILTER_OC]));
    }

    if (output_shape[OUTPUT_C] % groups != 0) {
        throw ngraph_error(
            "Output channels for GroupConvolutionBias not divisible by groups: channels=" +
            std::to_string(output_shape[OUTPUT_C]) + ", groups= " + std::to_string(groups));
    }
}

Shape op::GroupConvolutionBias::get_weights_dimensions() {
    // reshape weights into 5d tensors that includes groups
    const size_t OC = 0;
    const size_t OC_IN_OUTPUT = 1;
    const size_t IC = 1;

    Shape weights_shape_groups{get_inputs().at(1).get_shape()};

    weights_shape_groups.at(OC) = get_shape().at(OC_IN_OUTPUT) / get_groups();
    weights_shape_groups.at(IC) = get_inputs().at(0).get_shape().at(IC) / get_groups();

    // push_front the number of groups
    weights_shape_groups.insert(weights_shape_groups.begin(), get_groups());
    return weights_shape_groups;
}

op::GroupConvolutionBias::GroupConvolutionBias(const shared_ptr<op::GroupConvolution>& conv,
                                               const shared_ptr<Node>& bias,
                                               size_t groups,
                                               const Shape& output_shape,
                                               bool with_relu,
                                               float alpha)
    : Op("GroupConvolutionBias",
         check_single_output_args({conv->get_argument(0), conv->get_argument(1), bias}))
    , m_window_movement_strides(conv->get_window_movement_strides())
    , m_window_dilation_strides(conv->get_window_dilation_strides())
    , m_padding_below(conv->get_padding_below())
    , m_padding_above(conv->get_padding_above())
    , m_data_dilation_strides(conv->get_data_dilation_strides())
    , m_with_relu(with_relu)
    , m_groups(groups)
    , m_alpha(alpha) {
    constructor_validate_and_infer_types();

    if (conv->get_element_type() != bias->get_element_type()) {
        throw ngraph_error("GroupConvolution's element type isn't equal to bias!");
    }

    validate_groupconvbias_shapes(conv->get_argument(0)->get_shape(),
                                  conv->get_argument(1)->get_shape(),
                                  bias->get_shape(),
                                  output_shape,
                                  groups);

    set_output_type(0, conv->get_element_type(), output_shape);
}

op::GroupConvolutionBias::GroupConvolutionBias(const shared_ptr<Node>& data_batch,
                                               const shared_ptr<Node>& filters,
                                               const shared_ptr<Node>& bias,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               size_t groups,
                                               const Shape& output_shape,
                                               bool with_relu,
                                               float alpha)
    : Op("GroupConvolutionBias", check_single_output_args({data_batch, filters, bias}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
    , m_groups(groups)
    , m_alpha(alpha) {
    constructor_validate_and_infer_types();

    auto& data_batch_shape = data_batch->get_shape();
    auto& data_batch_et = data_batch->get_element_type();
    auto& filters_shape = filters->get_shape();
    auto& filters_et = filters->get_element_type();

    //
    //  Make sure data batch and filter element types match.
    //
    if (data_batch_et != filters_et) {
        throw ngraph_error("GroupConvolutionBias data batch and filter element types do not match");
    }

    validate_groupconvbias_shapes(
        data_batch_shape, filters_shape, bias->get_shape(), output_shape, groups);

    set_output_type(0, data_batch_et, output_shape);
}

shared_ptr<Node> op::GroupConvolutionBias::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 3) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return shared_ptr<Node>(new GroupConvolutionBias(new_args.at(0),
                                                     new_args.at(1),
                                                     new_args.at(2),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     get_groups(),
                                                     get_output_shape(0),
                                                     m_with_relu,
                                                     get_alpha()));
}

void op::GroupConvolutionBias::generate_adjoints(autodiff::Adjoints& adjoints,
                                                 const NodeVector& deltas) {
    throw ngraph_error("GroupConvolutionBias generate_adjoints not supported implemented");
}
