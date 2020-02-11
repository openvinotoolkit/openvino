// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_ie.hpp"
#include <ie_error.hpp>

#include <algorithm>
#include <memory>
#include <vector>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ConvolutionIE::type_info;

op::ConvolutionIE::ConvolutionIE(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const Shape& output_shape,
                                 const size_t& group,
                                 const PadType& auto_pad)
        : Op({data_batch, filters})
        , m_strides(strides)
        , m_dilations(dilations)
        , m_pads_begin(pads_begin)
        , m_pads_end(pads_end)
        , m_auto_pad(auto_pad)
        , m_group(group)
        , m_output_shape(output_shape) {
    constructor_validate_and_infer_types();
}

op::ConvolutionIE::ConvolutionIE(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Output<Node>& bias,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const Shape& output_shape,
                                 const size_t& group,
                                 const PadType& auto_pad)
        : Op({data_batch, filters, bias})
        , m_strides(strides)
        , m_dilations(dilations)
        , m_pads_begin(pads_begin)
        , m_pads_end(pads_end)
        , m_auto_pad(auto_pad)
        , m_group(group)
        , m_output_shape(output_shape) {
    constructor_validate_and_infer_types();
}

void op::ConvolutionIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), m_output_shape);
}

shared_ptr<Node> op::ConvolutionIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() == 2) {
        return make_shared<ConvolutionIE>(new_args.at(0),
                                          new_args.at(1),
                                          m_strides,
                                          m_pads_begin,
                                          m_pads_end,
                                          m_dilations,
                                          m_output_shape,
                                          m_group,
                                          m_auto_pad);
    } else {
        return make_shared<ConvolutionIE>(new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          m_strides,
                                          m_pads_begin,
                                          m_pads_end,
                                          m_dilations,
                                          m_output_shape,
                                          m_group,
                                          m_auto_pad);
    }
}

shared_ptr<Node> op::ConvolutionIE::copy(const OutputVector& new_args) const {
    if (new_args.size() == 2) {
        return make_shared<ConvolutionIE>(new_args.at(0),
                                          new_args.at(1),
                                          m_strides,
                                          m_pads_begin,
                                          m_pads_end,
                                          m_dilations,
                                          m_output_shape,
                                          m_group,
                                          m_auto_pad);
    } else {
        return make_shared<ConvolutionIE>(new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          m_strides,
                                          m_pads_begin,
                                          m_pads_end,
                                          m_dilations,
                                          m_output_shape,
                                          m_group,
                                          m_auto_pad);
    }
}
