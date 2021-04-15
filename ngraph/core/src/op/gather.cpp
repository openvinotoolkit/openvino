// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gather.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/op/util/gather_base.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::Gather, "Gather", 1, op::util::GatherBase);

op::v1::Gather::Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axes)
    : GatherBase(params, indices, axes)
{
    constructor_validate_and_infer_types();
}

void op::v1::Gather::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Gather_validate_and_infer_types);
    // according to Gather_1 specification can accept any input type,
    // validate_tensor_type is not needed
    op::util::GatherBase::common_validate_and_infer_pshape();
}

bool ngraph::op::v1::Gather::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Gather_visit_attributes);
    return true;
}

shared_ptr<Node> op::v1::Gather::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Gather>(new_args.at(0), new_args.at(1), new_args.at(2));
}

NGRAPH_RTTI_DEFINITION(op::v7::Gather, "Gather", 7, op::util::GatherBase);

op::v7::Gather::Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims)
    : GatherBase(data, indices, axis, batch_dims)
{
    constructor_validate_and_infer_types();
}

void op::v7::Gather::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v7_Gather_validate_and_infer_types);

    // according to Gather_7 specification for indices and axis only int32/int64 are allowed
    validate_tensor_type(this, "indices", get_input_element_type(1), {element::i32, element::i64});
    validate_tensor_type(this, "axis", get_input_element_type(2), {element::i32, element::i64});

    op::util::GatherBase::common_validate_and_infer_pshape();
}

bool ngraph::op::v7::Gather::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v7_Gather_visit_attributes);
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

shared_ptr<Node> op::v7::Gather::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v7_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Gather>(new_args.at(0), new_args.at(1), new_args.at(2), m_batch_dims);
}
