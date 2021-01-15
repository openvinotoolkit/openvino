//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64. Default is i64");

    if (buckets_pshape.is_static())
    {
        NODE_VALIDATION_CHECK(
            this, buckets_pshape.rank().compatible(1), "buckets input must be a 1D tensor");
    }

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
