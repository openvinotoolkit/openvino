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
#include <ops.hpp>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/fused/batch_to_space.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::BatchToSpace::type_info;

ngraph::op::v1::BatchToSpace::BatchToSpace(const ngraph::Output<ngraph::Node>& data,
                                           const ngraph::Output<ngraph::Node>& block_shape,
                                           const ngraph::Output<ngraph::Node>& crops_begin,
                                           const ngraph::Output<ngraph::Node>& crops_end)
    : FusedOp({data, block_shape, crops_begin, crops_end})
{
    constructor_validate_and_infer_types();
}

NodeVector op::v1::BatchToSpace::decompose_op() const
{
    auto data = input_value(0);
    auto block = input_value(1);
    auto crops_begin = input_value(2);
    auto crops_end = input_value(3);

    const auto& data_shape = data.get_shape();

    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 2),
                          "The data tensor with rank lower than 2 is not supported (data rank: ",
                          data_shape.size(),
                          ")");

    const auto block_const = as_type_ptr<op::Constant>(block.get_node_shared_ptr());
    const auto crops_begin_const = as_type_ptr<op::Constant>(crops_begin.get_node_shared_ptr());
    const auto crops_end_const = as_type_ptr<op::Constant>(crops_end.get_node_shared_ptr());

    vector<int64_t> block_values, crops_end_values;
    block_values = block_const->cast_vector<int64_t>();
    crops_end_values = crops_end_const->cast_vector<int64_t>();

    // First we have to disperse the data from batch, then rearrange them
    // so as appropriate chunks of data where close to their destination place.
    // Finally squeeze data from respective dimensions.
    vector<int64_t> dispersed_shape;
    int64_t b_dim_divider = 1;
    for (const auto& el : block_values)
    {
        NODE_VALIDATION_CHECK(this, el > 0, "block_shape values must be greater than 0");
        b_dim_divider *= el;
    }

    NODE_VALIDATION_CHECK(this,
                          data_shape.at(0) % b_dim_divider == 0,
                          "BatchToSpace: The input data's 'batch' axis size: ",
                          data_shape.at(0),
                          " must be a multiple of ",
                          " product of block_shape values: ",
                          b_dim_divider);

    //   note: B_0 is expected to be 1.
    //      x' = reshape(`data`, [B_1, ..., B_{N - 1}, batch / (B_1 * ... B_{N - 1}), D_1, D_2, ...,
    //      D_{N - 1}]),
    //      where B_i = block_shape[i]
    dispersed_shape.insert(dispersed_shape.begin(), block_values.begin() + 1, block_values.end());
    dispersed_shape.push_back(data_shape.at(0) / b_dim_divider);
    for (size_t i = 1; i < data_shape.size(); ++i)
    {
        dispersed_shape.push_back(data_shape.at(i));
    }

    const auto out_pattern_1 =
        op::Constant::create(element::i64, Shape{dispersed_shape.size()}, dispersed_shape);
    const bool special_zero = false;
    auto flat_node = make_shared<ngraph::op::v1::Reshape>(data, out_pattern_1, special_zero)
                         ->add_provenance_group_members_above({data});

    // calculate axes to transpose
    //      x'' = transpose(x', [N, N + 1, 0, N + 2, 1, ..., N + N - 1, N - 1])
    vector<size_t> axes_order{block_values.size() - 1};
    for (size_t i = 0; i < block_values.size() - 1; ++i)
    {
        axes_order.push_back(i + block_values.size());
        axes_order.push_back(i);
    }
    flat_node = builder::opset1::reorder_axes(flat_node, axes_order);

    //   x''' = reshape(x'', [batch / (B_1 * ... * B_{N - 1}), D_1 * B_1, D_2 * B_2, ... , D_{N - 1}
    //   * B_{N - 1}])
    vector<int64_t> squeezed_shape;
    squeezed_shape.push_back(data_shape.at(0) / b_dim_divider);
    for (size_t i = 1; i < block_values.size(); ++i)
    {
        squeezed_shape.push_back(data_shape.at(i) * block_values.at(i));
    }

    const auto out_pattern_2 =
        op::Constant::create(element::i64, Shape{squeezed_shape.size()}, squeezed_shape);
    flat_node = make_shared<ngraph::op::v1::Reshape>(flat_node, out_pattern_2, special_zero)
                    ->add_provenance_group_members_above({data});

    //    Crop the start and end of dimensions according to `crops_begin`, `crops_end` to produce
    //    the output of shape:
    //    note: `crops_begin[0], crops_end[0]` are expected to be 0.
    //    `y = [batch / (B_1 * ... * B_{N - 1}), crop(D_1 * B_1, crops_begin[1], crops_end[1]),
    //          crop(D_2 * B_2, crops_begin[2], crops_end[2]), ... ,
    //          crop(D_{N - 1} * B_{N - 1}, crops_begin[N - 1], crops_end[N - 1])]`
    vector<int64_t> upperbounds_values;
    auto flat_node_shape = flat_node->get_shape();
    for (size_t i = 0; i < flat_node_shape.size(); ++i)
    {
        upperbounds_values.push_back(flat_node_shape.at(i) - crops_end_values.at(i));
    }
    const auto upperbounds = op::Constant::create(
        crops_end.get_element_type(), Shape{upperbounds_values.size()}, upperbounds_values);

    vector<int64_t> begin_mask(data_shape.size(), 0);
    vector<int64_t> end_mask(data_shape.size(), 0);
    flat_node = make_shared<op::v1::StridedSlice>(
        flat_node, crops_begin_const, upperbounds, begin_mask, end_mask);
    return NodeVector{flat_node};
}

void ngraph::op::v1::BatchToSpace::pre_validate_and_infer_types()
{
    PartialShape data_pshape = get_input_partial_shape(0);

    auto data = input_value(0);
    auto block = input_value(1);
    auto crops_begin = input_value(2);
    auto crops_end = input_value(3);
    NGRAPH_CHECK(is_type<op::v0::Constant>(block.get_node_shared_ptr()),
                 "block_shape input node is expected to be a static constant");

    NGRAPH_CHECK(is_type<op::v0::Constant>(crops_begin.get_node_shared_ptr()),
                 "crops_begin input node is expected to be a static constant");

    NGRAPH_CHECK(is_type<op::v0::Constant>(crops_end.get_node_shared_ptr()),
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

std::shared_ptr<ngraph::Node>
    ngraph::op::v1::BatchToSpace::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<BatchToSpace>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::BatchToSpace::visit_attributes(ngraph::AttributeVisitor& visitor)
{
    return true;
}
