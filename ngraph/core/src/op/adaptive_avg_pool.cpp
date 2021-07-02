// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/adaptive_avg_pool.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/reference/adaptive_avg_pool.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

namespace adaptive_avg_pool
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        runtime::reference::adaptive_avg_pool(
            arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg0->get_shape(), out->get_shape());
        return true;
    }

    bool evaluate_adaptive_avg_pool(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        bool rc = true;
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_add, i8, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, i16, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, i32, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, i64, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, u8, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, u16, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, u32, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, u64, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, bf16, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, f16, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_add, f32, arg0, out);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace adaptive_avg_pool

NGRAPH_RTTI_DEFINITION(op::v8::AdaptiveAvgPool, "AdaptiveAvgPool", 8);

op::v8::AdaptiveAvgPool::AdaptiveAvgPool(const Output<Node>& data, const Output<Node>& output_shape)
    : Op({data, output_shape})
{
    constructor_validate_and_infer_types();
}

bool op::v8::AdaptiveAvgPool::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_AdaptiveAvgPool_visit_attributes);
    return true;
}

void op::v8::AdaptiveAvgPool::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v8_AdaptiveAvgPool_validate_and_infer_types);

    const PartialShape& data_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          data_shape.rank().compatible(3) || data_shape.rank().compatible(4) ||
                              data_shape.rank().compatible(5),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          data_shape);

    auto output_shape = PartialShape::dynamic(data_shape.rank());
    if (data_shape.rank().is_static())
    {
        if (data_shape[0].is_static())
        {
            output_shape[0] = data_shape[0]; // batch size
        }
        if (data_shape[1].is_static())
        {
            output_shape[1] = data_shape[1]; // channel size
        }
        if (const auto& const_output_shape = get_constant_from_source(input_value(1)))
        {
            auto output_spatial_shape = const_output_shape->cast_vector<int64_t>();
            NODE_VALIDATION_CHECK(this,
                                  (size_t)data_shape.rank().get_length() ==
                                      2 + output_spatial_shape.size(),
                                  "Output shape is not compatible with input data rank");
            int i = 2;
            for (auto& dim : output_spatial_shape)
            {
                output_shape[i++] = dim;
            }
        }
    }
    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v8::AdaptiveAvgPool::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_AdaptiveAvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v8::AdaptiveAvgPool>(new_args.at(0), new_args.at(1));
}

bool op::v8::AdaptiveAvgPool::evaluate(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v8_AdaptiveAvgPool_evaluate);
    return adaptive_avg_pool::evaluate_adaptive_avg_pool(inputs[0], outputs[0]);
}

bool op::v8::AdaptiveAvgPool::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v8_AdaptiveAvgPool_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}
