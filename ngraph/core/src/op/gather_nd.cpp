//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/gather_nd.hpp"
#include "ngraph/shape.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static int PARAMS = 0;
static int INDICES = 1;

constexpr NodeTypeInfo op::GatherND::type_info;

shared_ptr<Node> op::GatherND::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GatherND>(new_args.at(PARAMS), new_args.at(INDICES));
}

void op::GatherND::validate_and_infer_types()
{
    element::Type result_et = get_input_element_type(PARAMS);
    element::Type indices_et = get_input_element_type(INDICES);

    const PartialShape& params_shape = get_input_partial_shape(PARAMS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    NODE_VALIDATION_CHECK(this,
                          indices_shape.rank().is_dynamic() ||
                              indices_shape.rank().get_length() >= 1,
                          "indices rank is expected to be at least 1");

    NODE_VALIDATION_CHECK(this,
                          params_shape.rank().is_dynamic() || params_shape.rank().get_length() >= 1,
                          "params rank is expected to be at least 1");

    NODE_VALIDATION_CHECK(this,
                          params_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
                              indices_shape[indices_shape.rank().get_length() - 1].get_length() <=
                                  params_shape.rank().get_length(),
                          "last dimension of indices can be at most the rank of params");

    PartialShape result_shape;
    if (params_shape.rank().is_static() && indices_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(
            indices_shape.rank().get_length() - 1 + params_shape.rank().get_length() -
            indices_shape[indices_shape.rank().get_length() - 1].get_length());
        size_t i = 0;
        for (; i < indices_shape.rank().get_length() - 1; i++)
        {
            result_dims[i] = indices_shape[i];
        }
        for (size_t j = indices_shape[indices_shape.rank().get_length() - 1].get_length();
             j < params_shape.rank().get_length();
             i++, j++)
        {
            result_dims[i] = params_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}
