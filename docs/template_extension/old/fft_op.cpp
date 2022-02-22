// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [fft_op:implementation]
#include "fft_op.hpp"

using namespace TemplateExtension;

FFTOp::FFTOp(const ngraph::Output<ngraph::Node>& inp, bool _inverse) : Op({inp}) {
    constructor_validate_and_infer_types();
    inverse = _inverse;
}

void FFTOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}

std::shared_ptr<ngraph::Node> FFTOp::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<FFTOp>(new_args.at(0), inverse);
}

bool FFTOp::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("inverse", inverse);
    return true;
}
//! [fft_op:implementation]
