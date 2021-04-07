// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/bucketize.hpp"
#include "itt.hpp"

using namespace ngraph;
using namespace std;

constexpr NodeTypeInfo op::v3::Bucketize::type_info;

op::v3::Bucketize::Bucketize(const Output<Node>& data,
                             const Output<Node>& buckets,
                             const element::Type output_type,
                             const bool with_right_bound)
    : Op({data, buckets})
    , m_output_type(output_type)
    , m_with_right_bound(with_right_bound)
{
    constructor_validate_and_infer_types();
}

bool op::v3::Bucketize::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_Bucketize_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("with_right_bound", m_with_right_bound);
    return true;
}

void op::v3::Bucketize::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_Bucketize_validate_and_infer_types);
    const PartialShape& data_pshape = get_input_partial_shape(0);
    const PartialShape& buckets_pshape = get_input_partial_shape(1);

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

    NODE_VALIDATION_CHECK(this,
                          buckets_pshape.rank().compatible(1),
                          "Buckets input must be a 1D tensor. Got: ",
                          buckets_pshape);

    if (data_pshape.is_dynamic())
    {
        set_input_is_relevant_to_shape(0);
    }

    set_output_size(1);
    set_output_type(0, m_output_type, data_pshape);
}

shared_ptr<Node> op::v3::Bucketize::clone_with_new_inputs(const OutputVector& inputs) const
{
    NGRAPH_OP_SCOPE(v3_Bucketize_clone_with_new_inputs);
    check_new_args_count(this, inputs);

    return make_shared<v3::Bucketize>(
        inputs.at(0), inputs.at(1), m_output_type, m_with_right_bound);
}
