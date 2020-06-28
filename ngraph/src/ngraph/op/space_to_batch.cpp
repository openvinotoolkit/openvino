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
#include <cmath>
#include <cstddef>
#include <memory>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/space_to_batch.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::SpaceToBatch::type_info;

ngraph::op::v1::SpaceToBatch::SpaceToBatch(const ngraph::Output<ngraph::Node>& data,
                                           const ngraph::Output<ngraph::Node>& block_shape,
                                           const ngraph::Output<ngraph::Node>& pads_begin,
                                           const ngraph::Output<ngraph::Node>& pads_end)
    : Op({data, block_shape, pads_begin, pads_end})
{
    constructor_validate_and_infer_types();
}

void op::v1::SpaceToBatch::validate_and_infer_types()
{
    PartialShape data_pshape = get_input_partial_shape(0);
    const auto& data_type = get_input_element_type(0);
    const auto& block_shape_type = get_input_element_type(1);
    const auto& pads_begin_type = get_input_element_type(2);
    const auto& pads_end_type = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          block_shape_type.is_integral_number(),
                          "block_shape must be an integral number but got (",
                          block_shape_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_begin_type.is_integral_number(),
                          "crops_begin must be an integral number but got (",
                          pads_begin_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_end_type.is_integral_number(),
                          "crops_end must be an integral number but got (",
                          pads_end_type,
                          ").");

    auto data = input_value(0);
    auto block = input_value(1);
    auto pads_begin = input_value(2);
    auto pads_end = input_value(3);

    if (block.get_node_shared_ptr()->is_constant() &&
        pads_begin.get_node_shared_ptr()->is_constant() &&
        pads_end.get_node_shared_ptr()->is_constant() && data_pshape.is_static())
    {
        const auto& data_shape = data.get_shape();

        NODE_VALIDATION_CHECK(
            this,
            (data_shape.size() >= 2),
            "The data tensor with rank lower than 2 is not supported (data rank: ",
            data_shape.size(),
            ")");

        auto block_val = std::dynamic_pointer_cast<op::Constant>(block.get_node_shared_ptr())
                             ->cast_vector<int64_t>();
        auto pads_begin_val =
            std::dynamic_pointer_cast<op::Constant>(pads_begin.get_node_shared_ptr())
                ->cast_vector<int64_t>();
        auto pads_end_val = std::dynamic_pointer_cast<op::Constant>(pads_end.get_node_shared_ptr())
                                ->cast_vector<int64_t>();

        int64_t block_prod = 1;
        for (long idx : block_val)
            block_prod *= idx;

        Shape output_shape = {static_cast<size_t>(data_shape[0] * block_prod)};
        for (size_t idx = 1; idx < data_shape.size(); ++idx)
        {
            NODE_VALIDATION_CHECK(
                this, block_val.at(idx) > 0, "block_shape values must be greater than 0");
            NODE_VALIDATION_CHECK(
                this,
                (pads_begin_val.at(idx) + data_shape.at(idx) + pads_end_val.at(idx)) %
                        block_val.at(idx) ==
                    0,
                "The dimension on position: ",
                idx,
                " equal to: ",
                pads_begin_val.at(idx) + data_shape.at(idx) + pads_end_val.at(idx),
                " must be a multiple of block_values[i]: ",
                block_val.at(idx));
            output_shape.push_back(
                static_cast<size_t>(pads_begin_val[idx] + data_shape[idx] + pads_end_val[idx]) /
                block_val[idx]);
        }

        set_output_size(1);
        set_output_type(0, data_type, output_shape);
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic());
    }
}

std::shared_ptr<Node>
    ngraph::op::v1::SpaceToBatch::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SpaceToBatch>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::SpaceToBatch::visit_attributes(ngraph::AttributeVisitor& visitor)
{
    return true;
}
