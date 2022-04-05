// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [fft_op:header]
#pragma once

#include <ngraph/ngraph.hpp>

namespace TemplateExtension {

class FFTOp : public ngraph::op::Op {
public:
    OPENVINO_OP("FFT", "custom_opset");

    FFTOp() = default;
    FFTOp(const ngraph::Output<ngraph::Node>& inp, bool inverse);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    bool inverse;
};

}  // namespace TemplateExtension
//! [fft_op:header]
