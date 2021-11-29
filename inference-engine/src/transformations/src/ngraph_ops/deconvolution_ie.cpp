// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <vector>
#include <ngraph/ops.hpp>
#include "itt.hpp"

#include "ngraph_ops/deconvolution_ie.hpp"

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph_ops/type_relaxed.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DeconvolutionIE::type_info;

op::DeconvolutionIE::DeconvolutionIE(const Output<Node>& data,
                                     const Output<Node>& filters,
                                     const Strides& strides,
                                     const Strides& dilations,
                                     const CoordinateDiff& pads_begin,
                                     const CoordinateDiff& pads_end,
                                     const element::Type output_type,
                                     const size_t& group,
                                     const PadType& auto_pad,
                                     const CoordinateDiff& output_padding,
                                     const std::shared_ptr<Node> & output_shape)
        : Op({data, filters})
        , m_strides(strides)
        , m_dilations(dilations)
        , m_pads_begin(pads_begin)
        , m_pads_end(pads_end)
        , m_auto_pad(auto_pad)
        , m_group(group)
        , m_output_padding(output_padding)
        , m_output_shape(output_shape)
        , m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

op::DeconvolutionIE::DeconvolutionIE(const Output<Node>& data,
                                     const Output<Node>& filters,
                                     const Output<Node>& bias,
                                     const Strides& strides,
                                     const Strides& dilations,
                                     const CoordinateDiff& pads_begin,
                                     const CoordinateDiff& pads_end,
                                     const element::Type output_type,
                                     const size_t& group,
                                     const PadType& auto_pad,
                                     const CoordinateDiff& output_padding,
                                     const std::shared_ptr<Node> & output_shape)
        : Op({data, filters, bias})
        , m_strides(strides)
        , m_dilations(dilations)
        , m_pads_begin(pads_begin)
        , m_pads_end(pads_end)
        , m_auto_pad(auto_pad)
        , m_group(group)
        , m_output_padding(output_padding)
        , m_output_shape(output_shape)
        , m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void op::DeconvolutionIE::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(DeconvolutionIE_validate_and_infer_types);
    // To calculate output shape we use opset1::GroupConvolutionBackPropData
    // but before we need to reshape weights from I(G*O)YX to GIOYX
    auto weights = input_value(1);
    const auto weights_pshape = weights.get_partial_shape();
    if (weights_pshape.is_static()) {
        auto weights_shape = weights_pshape.to_shape();
        std::vector<int64_t> reshape_dims(3);
        reshape_dims[0] = m_group; // G
        reshape_dims[1] = weights_shape[0]; // I
        reshape_dims[2] = weights_shape[1] / m_group; // O
        reshape_dims.insert(reshape_dims.end(), weights_shape.begin() + 2, weights_shape.end());
        weights = std::make_shared<opset1::Reshape>(weights, opset1::Constant::create(element::i64, Shape{reshape_dims.size()}, reshape_dims), true);
    }
    Output<Node> conv;
    if (m_output_shape) {
        conv = std::make_shared<op::TypeRelaxed<opset1::GroupConvolutionBackpropData>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(input_value(0), element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(weights, element::f32).get(),
            m_output_shape,
            m_strides,
            m_pads_begin,
            m_pads_end,
            m_dilations,
            m_auto_pad,
            m_output_padding);
    } else {
        conv = std::make_shared<op::TypeRelaxed<opset1::GroupConvolutionBackpropData>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(input_value(0), element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(weights, element::f32).get(),
            m_strides,
            m_pads_begin,
            m_pads_end,
            m_dilations,
            m_auto_pad,
            m_output_padding);
    }
    set_output_type(0, m_output_type, conv.get_partial_shape());
}

shared_ptr<Node> op::DeconvolutionIE::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    INTERNAL_OP_SCOPE(DeconvolutionIE_clone_with_new_inputs);
    if (new_args.size() == 2) {
        return make_shared<DeconvolutionIE>(new_args.at(0),
                                            new_args.at(1),
                                            m_strides,
                                            m_dilations,
                                            m_pads_begin,
                                            m_pads_end,
                                            m_output_type,
                                            m_group,
                                            m_auto_pad,
                                            m_output_padding,
                                            m_output_shape);
    } else if (new_args.size() == 3) {
        return make_shared<DeconvolutionIE>(new_args.at(0),
                                            new_args.at(1),
                                            new_args.at(2),
                                            m_strides,
                                            m_dilations,
                                            m_pads_begin,
                                            m_pads_end,
                                            m_output_type,
                                            m_group,
                                            m_auto_pad,
                                            m_output_padding,
                                            m_output_shape);
    }
    throw ngraph::ngraph_error("Unexpected number of arguments");
}

bool op::DeconvolutionIE::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(DeconvolutionIE_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("group", m_group);
    return true;
}
