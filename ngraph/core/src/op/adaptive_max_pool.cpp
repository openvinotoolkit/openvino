// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/adaptive_max_pool.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::AdaptiveMaxPool, "AdaptiveMaxPool", 8);

op::v8::AdaptiveMaxPool::AdaptiveMaxPool(const Output<Node>& data,
                                         const Output<Node>& output_shape,
                                         const ngraph::element::Type& index_element_type)
    : Op({data, output_shape})
    , m_index_element_type{index_element_type}
{
    constructor_validate_and_infer_types();
}

bool op::v8::AdaptiveMaxPool::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_AdaptiveMaxPool_visit_attributes);
    visitor.on_attribute("index_element_type", m_index_element_type);
    return true;
}

void op::v8::AdaptiveMaxPool::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v8_AdaptiveMaxPool_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          m_index_element_type == element::i64 ||
                              m_index_element_type == element::i32,
                          "Index element type must be i32 or i64");

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
    set_output_type(1, m_index_element_type, output_shape);
}

shared_ptr<Node> op::v8::AdaptiveMaxPool::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_AdaptiveMaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v8::AdaptiveMaxPool>(new_args.at(0), new_args.at(1), m_index_element_type);
}
