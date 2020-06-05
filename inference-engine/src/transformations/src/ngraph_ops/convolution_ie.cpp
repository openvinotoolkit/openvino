// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/convolution_ie.hpp"

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
                                 const Strides& dilations,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const element::Type output_type,
                                 const size_t& group,
                                 const PadType& auto_pad)
        : Op({data_batch, filters})
        , m_strides(strides)
        , m_dilations(dilations)
        , m_pads_begin(pads_begin)
        , m_pads_end(pads_end)
        , m_auto_pad(auto_pad)
        , m_group(group)
        , m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

op::ConvolutionIE::ConvolutionIE(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Output<Node>& bias,
                                 const Strides& strides,
                                 const Strides& dilations,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const element::Type output_type,
                                 const size_t& group,
                                 const PadType& auto_pad)
        : Op({data_batch, filters, bias})
        , m_strides(strides)
        , m_dilations(dilations)
        , m_pads_begin(pads_begin)
        , m_pads_end(pads_end)
        , m_auto_pad(auto_pad)
        , m_group(group)
        , m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void op::ConvolutionIE::validate_and_infer_types() {
    PartialShape data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    PartialShape filters_shape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    PartialShape result_shape{PartialShape::dynamic()};

    // In case if number of groups greater than 1 and channel dimension is dynamic we can't calculate output shape
    if (m_group > 1) {
        if (data_batch_shape.rank().is_dynamic() || data_batch_shape[1].is_dynamic()) {
            set_output_type(0, m_output_type, result_shape);
            return;
        } else {
            // Update channel dimension according to groups count
            data_batch_shape[1] = data_batch_shape[1].get_length() / m_group;
        }
    }

    // we need to adjust filters_shape to reuse helpers for normal convolution
    if (filters_shape.is_static() && data_batch_shape.is_static()) {
        if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER) {
            m_pads_begin.clear();
            m_pads_end.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(data_batch_shape.to_shape(),
                               filter_shape,
                               m_strides,
                               m_dilations,
                               m_auto_pad,
                               m_pads_end,
                               m_pads_begin);
        }
    }

    result_shape = infer_convolution_forward(this,
                                             data_batch_shape,
                                             Strides(m_strides.size(), 1), // dummy data dilations
                                             m_pads_begin,
                                             m_pads_end,
                                             filters_shape,
                                             m_strides,
                                             m_dilations);

    set_output_type(0, m_output_type, result_shape);
}

shared_ptr<Node> op::ConvolutionIE::clone_with_new_inputs(const ngraph::OutputVector & new_args) const {
    if (new_args.size() == 2) {
        return make_shared<ConvolutionIE>(new_args.at(0),
                                          new_args.at(1),
                                          m_strides,
                                          m_dilations,
                                          m_pads_begin,
                                          m_pads_end,
                                          m_output_type,
                                          m_group,
                                          m_auto_pad);
    } else if (new_args.size() == 3) {
        return make_shared<ConvolutionIE>(new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          m_strides,
                                          m_dilations,
                                          m_pads_begin,
                                          m_pads_end,
                                          m_output_type,
                                          m_group,
                                          m_auto_pad);
    }

    throw ngraph_error("Unsupported number of arguments for ConvolutionIE operation");
}
