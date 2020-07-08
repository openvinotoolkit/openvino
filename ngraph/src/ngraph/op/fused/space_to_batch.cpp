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
#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/fused/space_to_batch.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::SpaceToBatch::type_info;

ngraph::op::v1::SpaceToBatch::SpaceToBatch(const ngraph::Output<ngraph::Node>& data,
                                           const ngraph::Output<ngraph::Node>& block_shape,
                                           const ngraph::Output<ngraph::Node>& pads_begin,
                                           const ngraph::Output<ngraph::Node>& pads_end)
    : FusedOp({data, block_shape, pads_begin, pads_end})
{
    constructor_validate_and_infer_types();
}

OutputVector op::v1::SpaceToBatch::decompose_op() const
{
    auto data = input_value(0);
    auto block = input_value(1);
    auto pads_begin = input_value(2);
    auto pads_end = input_value(3);

    const auto& data_shape = data.get_shape();

    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 2),
                          "The data tensor with rank lower than 2 is not supported (data rank: ",
                          data_shape.size(),
                          ")");

    const auto block_const = as_type_ptr<op::Constant>(block.get_node_shared_ptr());
    const auto pads_begin_const = as_type_ptr<op::Constant>(pads_begin.get_node_shared_ptr());
    const auto pads_end_const = as_type_ptr<op::Constant>(pads_end.get_node_shared_ptr());

    vector<int64_t> block_values;
    block_values = block_const->cast_vector<int64_t>();

    //    Zero-pad the start and end of dimensions [D_0, ..., D_{N - 1}] of the input according to
    //    `pads_begin`
    //    and `pads_end`:
    //    note: P_0 for batch dimension is expected to be 0 (no-padding).
    //      x = [batch + P_0, D_1 + P_1, D_2 + P_2, ..., D_{N - 1} + P_{N - 1}], where P_i =
    //      pads_begin[i] + pads_end[i]
    auto out = make_shared<op::v1::Pad>(data, pads_begin_const, pads_end_const, PadMode::CONSTANT);
    auto out_shape = out->get_shape();

    // First we have to disperse the data from spatial dimensions, then
    // rearrange them so as appropriate chunks of data where close to their
    // destination place. Finally squeeze data from respective dimensions.
    Shape dispersed_shape{out_shape.at(0)};

    //    note: B_0 for batch is ignored.
    //      x' = reshape(x, [batch, (D_1 + P_1) / B_1, B_1, (D_2 + P_2) / B_2, B_2, ...,
    //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}, B_{N - 1}]), where B_i = block_shape[i]
    for (size_t i = 1; i < block_values.size(); ++i)
    {
        NODE_VALIDATION_CHECK(
            this, block_values.at(i) > 0, "block_shape values must be greater than 0");
        NODE_VALIDATION_CHECK(this,
                              out_shape.at(i) % block_values.at(i) == 0,
                              "The dimension on position: ",
                              i,
                              " equal to: ",
                              out_shape.at(i),
                              " must be a multiple of block_values[i]: ",
                              block_values.at(i));
        dispersed_shape.push_back(out_shape.at(i) / block_values.at(i));
        dispersed_shape.push_back(block_values.at(i));
    }
    auto flat_node = builder::opset1::reshape(out, dispersed_shape);

    //    x'' = transpose(x',  [2, 4, ..., (N - 1) + (N - 1), 0, 1, 3, ..., N + (N - 1)])
    vector<size_t> axes_order;
    for (size_t i = 0, j = 2; i < block_values.size() - 1; ++i, j += 2)
    {
        axes_order.push_back(j);
    }
    axes_order.push_back(0);
    for (size_t i = 0, j = 1; i < block_values.size() - 1; ++i, j += 2)
    {
        axes_order.push_back(j);
    }

    flat_node = builder::opset1::reorder_axes(flat_node, axes_order);
    Shape squeezed_shape;
    int64_t prod = 1;
    for (const auto& el : block_values)
    {
        prod *= el;
    }

    //    y = reshape(x'', [batch * B_1 * ... * B_{N - 1}, (D_1 + P_1) / B_1, (D_2 + P_2) / B_2, ...
    //    ,
    //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}])
    squeezed_shape.push_back(out_shape.at(0) * prod);
    for (size_t i = 1; i < block_values.size(); ++i)
    {
        squeezed_shape.push_back(out_shape.at(i) / block_values.at(i));
    }
    flat_node = builder::opset1::reshape(flat_node, squeezed_shape);

    return OutputVector{flat_node};
}

void ngraph::op::v1::SpaceToBatch::pre_validate_and_infer_types()
{
    PartialShape data_pshape = get_input_partial_shape(0);

    auto data = input_value(0);
    auto block = input_value(1);
    auto crops_begin = input_value(2);
    auto crops_end = input_value(3);
    NGRAPH_CHECK(block.get_node_shared_ptr()->is_constant(),
                 "block_shape input node is expected to be a static constant");

    NGRAPH_CHECK(crops_begin.get_node_shared_ptr()->is_constant(),
                 "crops_begin input node is expected to be a static constant");

    NGRAPH_CHECK(crops_end.get_node_shared_ptr()->is_constant(),
                 "crops_end input node is expected to be a static constant");

    const auto& data_type = get_input_element_type(0);
    const auto& block_shape_type = get_input_element_type(1);
    const auto& crops_begin_type = get_input_element_type(2);
    const auto& crops_end_type = get_input_element_type(3);
    NODE_VALIDATION_CHECK(this,
                          block_shape_type.is_integral_number(),
                          "block_shape must be an integral number but got (",
                          block_shape_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          crops_begin_type.is_integral_number(),
                          "crops_begin must be an integral number but got (",
                          crops_begin_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          crops_end_type.is_integral_number(),
                          "crops_end must be an integral number but got (",
                          crops_end_type,
                          ").");

    if (data_pshape.is_dynamic())
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
