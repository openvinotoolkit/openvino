// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include "itt.hpp"

using namespace ov;

BWDCMP_RTTI_DEFINITION(op::v9::GridSample);

op::v9::GridSample::GridSample(const Output<Node>& data, const Output<Node>& grid, const Attributes& attributes) {
    constructor_validate_and_infer_types();
}

bool op::v9::GridSample::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v9_GridSample_visit_attributes);
    visitor.on_attribute("align_corners", m_attributes.align_corners);
    visitor.on_attribute("mode", m_attributes.mode);
    visitor.on_attribute("padding_mode", m_attributes.padding_mode);
    return true;
}

void op::v9::GridSample::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v9_GridSample_validate_and_infer_types);
}

std::shared_ptr<Node> op::v9::GridSample::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v9_GridSample_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v9::GridSample>(new_args.at(0), new_args.at(1), this->get_attributes());
}
