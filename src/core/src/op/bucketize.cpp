// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/bucketize.hpp"

#include "bucketize_shape_inference.hpp"
#include "itt.hpp"

using namespace ngraph;
using namespace std;

BWDCMP_RTTI_DEFINITION(op::v3::Bucketize);

op::v3::Bucketize::Bucketize(const Output<Node>& data,
                             const Output<Node>& buckets,
                             const element::Type output_type,
                             const bool with_right_bound)
    : Op({data, buckets}),
      m_output_type(output_type),
      m_with_right_bound(with_right_bound) {
    constructor_validate_and_infer_types();
}

bool op::v3::Bucketize::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_Bucketize_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("with_right_bound", m_with_right_bound);
    return true;
}

void op::v3::Bucketize::validate_and_infer_types() {
    OV_OP_SCOPE(v3_Bucketize_validate_and_infer_types);
    const ov::PartialShape& data_pshape = get_input_partial_shape(0);
    const ov::PartialShape& buckets_pshape = get_input_partial_shape(1);

    const auto data_et = get_input_element_type(0);
    const auto buckets_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          data_et.is_real() || data_et.is_integral_number(),
                          "Data input type must be numeric. Got: ",
                          data_et);

    NODE_VALIDATION_CHECK(this,
                          buckets_et.is_real() || buckets_et.is_integral_number(),
                          "Buckets input type must be numeric. Got: ",
                          buckets_et);

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64. Got: ",
                          m_output_type);

    std::vector<ov::PartialShape> input_shapes = {data_pshape, buckets_pshape};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape::dynamic()};
    shape_infer(this, input_shapes, output_shapes);

    if (data_pshape.is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }

    set_output_size(1);
    set_output_type(0, m_output_type, output_shapes[0]);
}

shared_ptr<Node> op::v3::Bucketize::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v3_Bucketize_clone_with_new_inputs);
    check_new_args_count(this, inputs);

    return make_shared<v3::Bucketize>(inputs.at(0), inputs.at(1), m_output_type, m_with_right_bound);
}
