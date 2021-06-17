// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/deformable_convolution.hpp"
#include "itt.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::DeformableConvolution, "DeformableConvolution", 1, op::util::DeformableConvolutionBase);
NGRAPH_RTTI_DEFINITION(op::v8::DeformableConvolution, "DeformableConvolution", 8, op::util::DeformableConvolutionBase);

op::v8::DeformableConvolution::DeformableConvolution(const Output<Node> &arg,
                                                     const Output<Node> &offsets,
                                                     const Output<Node> &filters,
                                                     const Strides &strides,
                                                     const CoordinateDiff &pads_begin,
                                                     const CoordinateDiff &pads_end,
                                                     const Strides &dilations,
                                                     const op::PadType &auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group)
        : DeformableConvolutionBase(arg, offsets, filters, strides, pads_begin, pads_end, dilations, auto_pad, group,
                                    deformable_group) {

}

bool op::v8::DeformableConvolution::visit_attributes(AttributeVisitor &visitor) {
    return DeformableConvolutionBase::visit_attributes(visitor);
}

void op::v8::DeformableConvolution::validate_and_infer_types() {
    DeformableConvolutionBase::validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::DeformableConvolution::clone_with_new_inputs(const OutputVector &new_args) const {
    return DeformableConvolutionBase::clone_with_new_inputs(new_args);
}

namespace deformable_convolution {
    bool evaluate_deformable_convolution() {
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto offset_data_ptr = inputs[1]->get_data_ptr<ET>();
        const auto filter_data_ptr = inputs[2]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& offset_shape = inputs[1]->get_shape();
        const auto& filter_shape = inputs[2]->get_shape();
        runtime::reference::deformable_convolution<typename element_type_traits<ET>::value_type>(
                in_data_ptr,
                offset_data_ptr,
                filter_data_ptr,
                out_data_ptr,
                in_shape,
                offset_shape,
                filter_shape,
                out_shape,
                get_strides(),
                get_dilations(),
                get_pads_begin(),
                get_pads_end(),
                get_group(),
                get_deformable_group());
    }
}
bool op::v8::DeformableConvolution::evaluate(const HostTensorVector &outputs, const HostTensorVector &inputs) const {
    NGRAPH_OP_SCOPE(v8_DeformableConvolution);

    return true;
}


op::v1::DeformableConvolution::DeformableConvolution(const Output<Node> &arg,
                                                     const Output<Node> &offsets,
                                                     const Output<Node> &filters,
                                                     const Strides &strides,
                                                     const CoordinateDiff &pads_begin,
                                                     const CoordinateDiff &pads_end,
                                                     const Strides &dilations,
                                                     const op::PadType &auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group)
        : DeformableConvolutionBase(arg, offsets, filters, strides, pads_begin, pads_end, dilations, auto_pad, group,
                                    deformable_group) {

}
