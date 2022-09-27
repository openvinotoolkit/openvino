// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reorg_yolo.hpp"

#include "itt.hpp"
#include "ngraph/runtime/reference/reorg_yolo.hpp"
#include "reorg_yolo_shape_inference.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::ReorgYolo);

op::ReorgYolo::ReorgYolo(const Output<Node>& input, const Strides& strides) : Op({input}), m_strides(strides) {
    constructor_validate_and_infer_types();
}

op::ReorgYolo::ReorgYolo(const Output<Node>& input, const size_t stride)
    : Op({input}),
      m_strides(std::vector<size_t>{stride, stride}) {
    constructor_validate_and_infer_types();
}

void op::ReorgYolo::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ReorgYolo_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, !m_strides.empty(), "Stride attribute is required.");

    auto input_et = get_input_element_type(0);

    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    shape_infer(this, input_shapes, output_shapes);

    set_output_type(0, input_et, output_shapes[0]);
}

shared_ptr<Node> op::ReorgYolo::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ReorgYolo_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReorgYolo>(new_args.at(0), m_strides);
}

bool op::ReorgYolo::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ReorgYolo_visit_attributes);
    visitor.on_attribute("stride", m_strides);
    return true;
}
