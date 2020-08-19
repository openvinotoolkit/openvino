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

#include <memory>
#include <numeric>
#include <set>

#include "transformations/common_optimizations/algebraic_simplification.hpp"

#include <ngraph/log.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

using namespace std;
using namespace ngraph;

//`simplify_gather`, optimizes gather if Gather is gathering the
// whole input tensor
static bool simplify_gather(std::shared_ptr<Node> node) {
    if (auto gather = as_type_ptr<opset3::Gather>(node)) {
        // check if we are gathering the whole input
        auto data = gather->input_value(0);
        auto indices = gather->input_value(1);

        // we need to know data and indices shape to infer if gather is Nop
        if (data.get_partial_shape().is_dynamic() || indices.get_partial_shape().is_dynamic()) {
            return false;
        }
        // if rank of data and gather output dont match, we will skip
        if (data.get_shape().size() != node->get_shape().size()) {
            return false;
        }

        auto axis = gather->get_axis();

        if (axis == opset3::Gather::AXIS_NOT_SET_VALUE) {
            NGRAPH_DEBUG << "axis value not set";
            return false;
        }

        // case_1 : if the input tensor is of shape (4, 1, 4)
        // and axis = 1, then the gather would be simply
        // gathering the whole input tensor, so we can optimize this
        // op has Nop

        if (data.get_shape()[axis] == 1 && data.get_shape() == node->get_shape()) {
            return replace_output_update_name(gather->output(0), gather->input_value(0));
        }

        // case_2 : if the input tensor is of shape (4, 3, 4)
        // we need to check the contents of indices, if indices
        // is 1D tensor of value {0, 1, 2}, we can optimize this
        // op has Nop

        // check if the indices is constant
        auto constant_indices =
            as_type_ptr<opset3::Constant>(gather->input_value(1).get_node_shared_ptr());
        if (!constant_indices) {
            return false;
        } else {
            // if ref_inidices == indices, we are capturing the
            // entire input tensor
            std::vector<int64_t> ref_indices(data.get_shape()[axis], 0);
            std::iota(ref_indices.begin(), ref_indices.end(), 0);
            if (ref_indices == constant_indices->cast_vector<int64_t>()) {
                return replace_output_update_name(gather->output(0), gather->input_value(0));
            }
        }
    }
    return false;
}

// optimizes `gather->shapeof` into `shapeof->gather` for 0D indices
// other cases into Concat of shapeof/gather(data) + shapeof(indices)
static bool simplify_gather_shapeof(shared_ptr<Node> node) {
    auto gather = as_type_ptr<opset3::Gather>(node->input_value(0).get_node_shared_ptr());
    if (!gather) {
        return false;
    }
    auto gather_in_rank = gather->get_input_partial_shape(0).rank();
    auto indices_rank = gather->get_input_partial_shape(1).rank();
    auto axis = gather->get_axis();
    if (gather_in_rank.is_dynamic() || indices_rank.is_dynamic() ||
        axis == opset3::Gather::AXIS_NOT_SET_VALUE) {
        NGRAPH_DEBUG << gather << " cannot simplify gather->shapeof";
        return false;
    }

    auto zero_axis = opset3::Constant::create<int64_t>(element::i64, Shape{}, {0});
    NodeVector new_ops;
    auto new_shapeof = make_shared<opset3::ShapeOf>(gather->input_value(0));
    new_ops.push_back(new_shapeof);
    std::shared_ptr<Node> replace_op;
    if (indices_rank.get_length() == 0) {
        std::vector<int64_t> vi(gather_in_rank.get_length());
        std::iota(vi.begin(), vi.end(), 0);
        vi.erase(vi.begin() + axis);
        auto new_indices = opset3::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
        replace_op = make_shared<opset3::Gather>(new_shapeof, new_indices, zero_axis);
        new_ops.push_back(replace_op);
    } else {
        NodeVector concat_inputs;
        if (axis > 0) {
            std::vector<int64_t> vi(axis);
            std::iota(vi.begin(), vi.end(), 0);
            auto indices = opset3::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
            auto gather = make_shared<opset3::Gather>(new_shapeof, indices, zero_axis);
            new_ops.push_back(gather);
            concat_inputs.push_back(gather);
        }
        auto shapeof_indices = make_shared<opset3::ShapeOf>(gather->input_value(1));
        new_ops.push_back(shapeof_indices);

        concat_inputs.push_back(shapeof_indices);

        if (gather_in_rank.get_length() - 1 > axis) {
            std::vector<int64_t> vi(gather_in_rank.get_length() - (axis + 1));
            std::iota(vi.begin(), vi.end(), axis + 1);
            auto indices = opset3::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
            auto gather = make_shared<opset3::Gather>(new_shapeof, indices, zero_axis);
            new_ops.push_back(gather);
            concat_inputs.push_back(gather);
        }
        replace_op = make_shared<opset3::Concat>(concat_inputs, 0);
        new_ops.push_back(replace_op);
    }
    replace_op->set_friendly_name(node->get_friendly_name());
    copy_runtime_info(node, new_ops);
    replace_node(node, replace_op);
    return true;
}

static bool replace_transpose_with_reshape(shared_ptr<Node> transpose) {
    auto data = transpose->input_value(0);
    const auto input_shape = transpose->input(0).get_partial_shape();
    if (input_shape.rank().is_dynamic()) {
        return false;
    }

    const auto input_shape_rank = input_shape.rank().get_length();

    auto order = as_type_ptr<opset3::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!order) {
        return false;
    }

    const auto order_value = order->cast_vector<int64_t>();

    // Check that transpose order without 1 dims has an ascending order
    int64_t last_dim(-1);
    for (size_t i = 0; i < input_shape_rank; ++i) {
        if (input_shape[order_value[i]].is_dynamic() || input_shape[order_value[i]] != 1) {
            if (order_value[i] < last_dim) {
                return false;
            }
            last_dim = order_value[i];
        }
    }

    // Transpose operation can be removed if original transpose order is sorted
    // or dimension that changes their places equal to 1
    using DimensionToPosition = struct {
        Dimension dim;
        size_t pos;
    };
    std::vector<DimensionToPosition> dims;
    for (size_t i = 0; i < input_shape_rank; ++i) {
        if (order_value[i] != i) {
            dims.push_back({input_shape[order_value[i]], i});
        }
    }

    // If number of dimensions != 1 to move equal to 0 we can remove this Transpose
    if (count_if(dims.begin(), dims.end(), [](const DimensionToPosition& item) {
            return !(item.dim.is_static() && item.dim.get_length() == 1);
        }) == 0) {
        return replace_output_update_name(transpose->output(0), transpose->input_value(0));
    }

    // Transpose can be replaced with Reshape in two ways:
    // 1. Reshape with dims as Constant
    // 2. Reshape with dims as input (ShapeOf->Gather)
    //
    // The first case is possible only if one or less dynamic dimensions changes their position
    // For example: input_shape {?, 3, 1, ?} and order {0, 1, 3, 2} can be replaced with Reshape
    // with Constant {0, 3, -1, 1} but if input_shape {?, 1, 1, ?} and order {1, 0, 3, 2} transpose
    // cannot be replaced int the same way and in this case its only possible to use Gather(ShapeOf,
    // order)

    Output<Node> reshape_dim;
    NodeVector new_ops;

    if (count_if(dims.begin(), dims.end(), [](const DimensionToPosition& item) {
            return item.dim.is_dynamic();
        }) < 2) {
        vector<int64_t> reshape_value(input_shape_rank, 0);
        for (const auto& item : dims) {
            reshape_value[item.pos] = item.dim.is_dynamic() ? -1 : item.dim.get_length();
        }
        reshape_dim =
            opset3::Constant::create(element::i64, Shape{reshape_value.size()}, reshape_value);
    } else {
        auto shape_of = make_shared<opset3::ShapeOf>(data);
        new_ops.push_back(shape_of);
        reshape_dim = make_shared<opset3::Gather>(
            shape_of, order, opset3::Constant::create(element::i64, Shape{1}, {0}));
        new_ops.push_back(reshape_dim.get_node_shared_ptr());
    }

    auto reshape_op = make_shared<opset3::Reshape>(data, reshape_dim, true);
    new_ops.push_back(reshape_op);

    reshape_op->set_friendly_name(transpose->get_friendly_name());
    copy_runtime_info(transpose, new_ops);
    replace_node(transpose, reshape_op);

    return true;
}

bool pass::AlgebraicSimplification::run_on_function(shared_ptr<Function> f) {
    static const unordered_map<NodeTypeInfo, function<bool(shared_ptr<Node>)>> ops_to_simplifiers =
        {{opset3::Gather::type_info, simplify_gather},
         {opset2::ShapeOf::type_info, simplify_gather_shapeof},
         {opset3::ShapeOf::type_info, simplify_gather_shapeof},
         {opset3::Transpose::type_info, replace_transpose_with_reshape}};

    bool replaced = false;
    for (auto n : f->get_ordered_ops()) {
        if (op::is_output(n) || op::is_parameter(n)) {
            continue;
        }

        auto eh = ops_to_simplifiers.find(n->get_type_info());
        if (eh != ops_to_simplifiers.end()) {
            replaced = eh->second(n) || replaced;
        }
    }
    return replaced;
}
