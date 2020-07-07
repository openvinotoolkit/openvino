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

#include "algebraic_simplification.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/rt_info.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static shared_ptr<pattern::Matcher>
    create_binary_matcher(shared_ptr<pattern::op::Label> label,
                          shared_ptr<pattern::op::Label> const_label)
{
    auto bcst =
        make_shared<pattern::op::Skip>(const_label, pattern::has_class<op::v0::Broadcast>());
    auto bcst_label = make_shared<pattern::op::Label>(bcst, nullptr, NodeVector{bcst});
    auto matcher = make_shared<pattern::Matcher>(make_shared<T>(label, bcst_label));
    return matcher;
}

//`simplify_concat` identifies slices-concat sequences
// that cancel each other. Namely it replaces subgraphs
// similar to the one below with `arg`
//
//                 +----------+
//            +----+slice(n/2..n)---+
// +-------+  |    +----------+    |  +-----------+
// |  arg  +--+                    +--+  concat   |
// +-------+  |    +----------+    |  +-----------+
//            +----+slice(0..n/2)---+
//                 +----------+
static bool simplify_concat(shared_ptr<Node> n)
{
    NGRAPH_DEBUG << "In simplify_concat for " << n->get_name();
    if (n->get_output_partial_shape(0).is_dynamic())
    {
        NGRAPH_DEBUG << n << " has dynamic shape";
        return false;
    }

    Output<Node> branch_tip;

    auto ltip = make_shared<pattern::op::Label>(element::i32, Shape{2, 1});

    auto pslice =
        make_shared<op::v0::Slice>(ltip, Coordinate{0, 0}, Coordinate{2, 1}, Strides{1, 1});

    auto lslice = make_shared<pattern::op::Label>(pslice, nullptr, NodeVector{pslice});

    auto skip_reshape =
        make_shared<pattern::op::Skip>(lslice, pattern::has_class<op::v0::Reshape>());

    auto matcher = make_shared<pattern::Matcher>(skip_reshape);

    Coordinate prev_lower_bounds;
    Shape prev_slice_shape;

    for (auto carg : n->input_values())
    {
        if (!matcher->match(carg))
        {
            NGRAPH_DEBUG << carg << " doesn't match";
            return false;
        }

        auto& pattern_value_map = matcher->get_pattern_value_map();
        auto slice = as_type_ptr<op::v0::Slice>(pattern_value_map[lslice].get_node_shared_ptr());
        if (branch_tip != Output<Node>())
        {
            if (branch_tip != pattern_value_map[ltip])
            {
                NGRAPH_DEBUG << branch_tip << " doesn't match " << pattern_value_map[ltip];
                return false;
            }

            // slice chunks should be slice in the same order as slice nodes in concat's argument
            // list
            auto cur_lower_bounds = slice->get_lower_bounds();
            if (cur_lower_bounds < prev_lower_bounds)
            {
                NGRAPH_DEBUG << slice << " is in the wrong order";
                return false;
            }
            prev_lower_bounds.assign(cur_lower_bounds.begin(), cur_lower_bounds.end());

            // slice shapes need to match
            if (slice->get_shape() != prev_slice_shape)
            {
                NGRAPH_DEBUG << slice << " doesn't match the shape of the previous slice";
                return false;
            }
        }
        else
        {
            branch_tip = pattern_value_map[ltip];
            prev_lower_bounds.assign(slice->get_lower_bounds().begin(),
                                     slice->get_lower_bounds().end());
            prev_slice_shape.assign(slice->get_shape().begin(), slice->get_shape().end());
            NGRAPH_DEBUG << "setting branch_tip to " << branch_tip;
        }

        if (slice->get_users(true).size() > 1)
        {
            NGRAPH_DEBUG << slice << " has more than one user";
            return false;
        }

        if (shape_size(slice->get_strides()) != 1)
        {
            NGRAPH_DEBUG << slice << " is strided";
            return false;
        }

        // check that no other node uses slices and reshapes
        if (auto rcarg = as_type_ptr<op::v0::Reshape>(carg.get_node_shared_ptr()))
        {
            auto default_shape = get_default_order(rcarg->input_value(0).get_shape());
            if (default_shape != rcarg->get_input_order())
            {
                NGRAPH_DEBUG << carg << " reshape also does transposes";
                return false;
            }

            if (rcarg->get_users(true).size() > 1)
            {
                NGRAPH_DEBUG << rcarg << " has more than one user";
                return false;
            }
        }
    }

    auto concat = static_pointer_cast<op::v0::Concat>(n);
    auto concat_axis = concat->get_concatenation_axis();

    auto slice_shape = branch_tip.get_node_shared_ptr()->get_users(true).at(0)->get_shape();
    size_t slice_axis = numeric_limits<size_t>::max();

    auto btip_shape = branch_tip.get_shape();

    // slices should cover all elements
    if (shape_size(btip_shape) != shape_size(n->get_shape()))
    {
        NGRAPH_DEBUG << "The number of elements in Concat (" << shape_size(n->get_shape())
                     << ")  and the total of elements in slices (" << shape_size(btip_shape)
                     << ") don't match";
        return false;
    }

    for (size_t i = 0; i < btip_shape.size(); i++)
    {
        if (btip_shape[i] != slice_shape[i])
        {
            if (slice_axis != numeric_limits<size_t>::max())
            {
                // multi-axis slice + concat do not cancel
                return false;
            }
            slice_axis = i;
        }
    }

    if (slice_axis == numeric_limits<size_t>::max())
    {
        return false;
    }
    auto replacement = branch_tip;
    if (btip_shape != n->get_shape())
    {
        auto default_order = get_default_order(btip_shape);
        if (concat_axis == slice_axis)
        {
            // logical reshape only
            replacement =
                make_shared<op::v0::Reshape>(branch_tip, default_order, concat->get_shape());
        }
        else
        {
            // axis reordering required
            auto transposed_shape = n->get_shape();

            if (btip_shape.size() >= transposed_shape.size())
            {
                AxisVector order = get_default_order(btip_shape);
                auto ax = order[slice_axis];
                order[slice_axis] = order[concat_axis];
                order[concat_axis] = ax;
                replacement = make_shared<op::v0::Reshape>(branch_tip, order, transposed_shape);
            }
            else if (btip_shape.size() < transposed_shape.size())
            {
                // intermediate logical reshape
                AxisVector order = get_default_order(transposed_shape);
                auto ax = order[slice_axis];
                order[slice_axis] = order[concat_axis];
                order[concat_axis] = ax;
                auto output_shape = apply_permutation(transposed_shape, order);
                auto logical_reshape =
                    make_shared<op::v0::Reshape>(branch_tip, default_order, output_shape);
                // transpose to final concatenated shape
                replacement =
                    make_shared<op::v0::Reshape>(logical_reshape, order, transposed_shape);
            }
        }
    }
    n->output(0).replace(replacement);
    return true;
}

static bool is_uniform_constant(const op::Constant* constant, int value)
{
    bool rc = false;
    if (constant && constant->get_all_data_elements_bitwise_identical())
    {
        switch (constant->get_element_type())
        {
        case ngraph::element::Type_t::undefined:
        {
            throw runtime_error("is_value type not supported");
        }
        case ngraph::element::Type_t::dynamic: { throw runtime_error("is_value type not supported");
        }
        case ngraph::element::Type_t::boolean: break;
        case ngraph::element::Type_t::bf16:
            rc = *static_cast<const bfloat16*>(constant->get_data_ptr()) ==
                 bfloat16(static_cast<float>(value));
            break;
        case ngraph::element::Type_t::f16:
            rc = *static_cast<const float16*>(constant->get_data_ptr()) ==
                 float16(static_cast<float>(value));
            break;
        case ngraph::element::Type_t::f32:
            rc = *static_cast<const float*>(constant->get_data_ptr()) == static_cast<float>(value);
            break;
        case ngraph::element::Type_t::f64:
            rc =
                *static_cast<const double*>(constant->get_data_ptr()) == static_cast<double>(value);
            break;
        case ngraph::element::Type_t::i8:
            rc =
                *static_cast<const int8_t*>(constant->get_data_ptr()) == static_cast<int8_t>(value);
            break;
        case ngraph::element::Type_t::i16:
            rc = *static_cast<const int16_t*>(constant->get_data_ptr()) ==
                 static_cast<int16_t>(value);
            break;
        case ngraph::element::Type_t::i32:
            rc = *static_cast<const int32_t*>(constant->get_data_ptr()) ==
                 static_cast<int32_t>(value);
            break;
        case ngraph::element::Type_t::i64:
            rc = *static_cast<const int64_t*>(constant->get_data_ptr()) ==
                 static_cast<int64_t>(value);
            break;
        case ngraph::element::Type_t::u1: throw runtime_error("is_value type not supported");
        case ngraph::element::Type_t::u8:
            rc = *static_cast<const uint8_t*>(constant->get_data_ptr()) ==
                 static_cast<uint8_t>(value);
            break;
        case ngraph::element::Type_t::u16:
            rc = *static_cast<const uint16_t*>(constant->get_data_ptr()) ==
                 static_cast<uint16_t>(value);
            break;
        case ngraph::element::Type_t::u32:
            rc = *static_cast<const uint32_t*>(constant->get_data_ptr()) ==
                 static_cast<uint32_t>(value);
            break;
        case ngraph::element::Type_t::u64:
            rc = *static_cast<const uint64_t*>(constant->get_data_ptr()) ==
                 static_cast<uint64_t>(value);
            break;
        }
    }
    return rc;
}

static shared_ptr<op::Constant> get_constant(shared_ptr<Node> op)
{
    set<Node::type_info_t> nomath = {op::v0::Broadcast::type_info,
                                     op::v0::Reshape::type_info,
                                     op::v1::Broadcast::type_info,
                                     opset3::Broadcast::type_info,
                                     opset3::Reshape::type_info};
    ;
    while (nomath.find(op->get_type_info()) != nomath.end())
    {
        op = op->get_input_node_shared_ptr(0);
    }
    return as_type_ptr<op::Constant>(op);
}

static bool is_input_uniform_constant(shared_ptr<Node> op,
                                      int constant_value,
                                      shared_ptr<Node>& constant,
                                      Output<Node>& value)
{
    bool rc = false;
    auto c = get_constant(op->get_input_node_shared_ptr(0));
    if (is_uniform_constant(c.get(), constant_value))
    {
        constant = op->get_input_node_shared_ptr(0);
        value = op->input_value(1);
        rc = true;
    }
    else
    {
        c = get_constant(op->get_input_node_shared_ptr(1));
        if (is_uniform_constant(c.get(), constant_value))
        {
            constant = op->get_input_node_shared_ptr(1);
            value = op->input_value(0);
            rc = true;
        }
    }
    return rc;
}

//`simplify_gather`, optimizes gather if Gather is gathering the
// whole input tensor
static bool simplify_gather(std::shared_ptr<Node> node)
{
    if (auto gather = as_type_ptr<opset3::Gather>(node))
    {
        // check if we are gathering the whole input
        auto data = gather->input_value(0);
        auto indices = gather->input_value(1);

        // we need to know data and indices shape to infer if gather is Nop
        if (data.get_partial_shape().is_dynamic() || indices.get_partial_shape().is_dynamic())
        {
            return false;
        }
        // if rank of data and gather output dont match, we will skip
        if (data.get_shape().size() != node->get_shape().size())
        {
            return false;
        }

        auto axis = gather->get_axis();

        if (axis == opset3::Gather::AXIS_NOT_SET_VALUE)
        {
            NGRAPH_DEBUG << "axis value not set";
            return false;
        }

        // case_1 : if the input tensor is of shape (4, 1, 4)
        // and axis = 1, then the gather would be simply
        // gathering the whole input tensor, so we can optimize this
        // op has Nop

        if (data.get_shape()[axis] == 1 && data.get_shape() == node->get_shape())
        {
            return replace_output_update_name(gather->output(0), gather->input_value(0));
        }

        // case_2 : if the input tensor is of shape (4, 3, 4)
        // we need to check the contents of indices, if indices
        // is 1D tensor of value {0, 1, 2}, we can optimize this
        // op has Nop

        // check if the indices is constant
        auto constant_indices =
            as_type_ptr<opset3::Constant>(gather->input_value(1).get_node_shared_ptr());
        if (!constant_indices)
        {
            return false;
        }
        else
        {
            // if ref_inidices == indices, we are capturing the
            // entire input tensor
            std::vector<int64_t> ref_indices(data.get_shape()[axis], 0);
            std::iota(ref_indices.begin(), ref_indices.end(), 0);
            if (ref_indices == constant_indices->cast_vector<int64_t>())
            {
                return replace_output_update_name(gather->output(0), gather->input_value(0));
            }
        }
    }
    return false;
}

//`simplify_multiply` optimizes the following 4 *base* cases
//(8 cases in total including variants due to commutativity)
//
// a * 0 -> 0
// a * broadcast(0) -> broadcast(0)
// a * 1 -> a
// a * broadcast(1) -> a
static bool simplify_multiply(shared_ptr<Node> multiply)
{
    bool rc = false;
    if (multiply)
    {
        shared_ptr<Node> constant;
        Output<Node> value;
        if (is_input_uniform_constant(multiply, 0, constant, value))
        {
            replace_output_update_name(multiply->output(0), constant->output(0));
            rc = true;
        }
        else
        {
            if (is_input_uniform_constant(multiply, 1, constant, value))
            {
                replace_output_update_name(multiply->output(0), value);
                rc = true;
            }
        }
    }

    return rc;
}

//`simplify_add` optimizes the following 2 *base* cases
//(4 cases in total including variants due to commutativity)
//
// a + 0 -> a
// a + broadcast(0) -> a
static bool simplify_add(shared_ptr<Node> add)
{
    bool rc = false;
    if (add)
    {
        shared_ptr<Node> constant;
        Output<Node> value;
        if (is_input_uniform_constant(add, 0, constant, value))
        {
            replace_output_update_name(add->output(0), value);
            rc = true;
        }
    }

    return rc;
}

//`simplify_log` optimizes `log(exp(x)/y)` into `x - log(y)`
static bool simplify_log(shared_ptr<Node> n)
{
    if (auto div = as_type_ptr<op::v0::Divide>(n->input_value(0).get_node_shared_ptr()))
    {
        if (auto exp = as_type_ptr<op::v0::Exp>(div->input_value(0).get_node_shared_ptr()))
        {
            auto denom = div->get_input_source_output(1);
            auto diff = make_shared<op::v0::Subtract>(exp->get_input_source_output(0),
                                                      make_shared<op::v0::Log>(denom));
            replace_node(n, diff);
            return true;
        }
    }

    return false;
}

// optimizes `gather->shapeof` into `shapeof->gather` for 0D indices
// other cases into Concat of shapeof/gather(data) + shapeof(indices)
static bool simplify_gather_shapeof(shared_ptr<Node> node)
{
    auto gather = as_type_ptr<opset3::Gather>(node->input_value(0).get_node_shared_ptr());
    if (!gather)
    {
        return false;
    }
    auto gather_in_rank = gather->get_input_partial_shape(0).rank();
    auto indices_rank = gather->get_input_partial_shape(1).rank();
    auto axis = gather->get_axis();
    if (gather_in_rank.is_dynamic() || indices_rank.is_dynamic() ||
        axis == opset3::Gather::AXIS_NOT_SET_VALUE)
    {
        NGRAPH_DEBUG << gather << " cannot simplify gather->shapeof";
        return false;
    }

    auto zero_axis = opset3::Constant::create<int64_t>(element::i64, Shape{}, {0});
    NodeVector new_ops;
    auto new_shapeof = make_shared<opset3::ShapeOf>(gather->input_value(0));
    new_ops.push_back(new_shapeof);
    std::shared_ptr<Node> replace_op;
    if (indices_rank.get_length() == 0)
    {
        std::vector<int64_t> vi(gather_in_rank.get_length());
        std::iota(vi.begin(), vi.end(), 0);
        vi.erase(vi.begin() + axis);
        auto new_indices = opset3::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
        replace_op = make_shared<opset3::Gather>(new_shapeof, new_indices, zero_axis);
        new_ops.push_back(replace_op);
    }
    else
    {
        NodeVector concat_inputs;
        if (axis > 0)
        {
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

        if (gather_in_rank.get_length() - 1 > axis)
        {
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

static size_t reduction_shape_size(const AxisSet& axes, const Shape& shape)
{
    size_t prod = 1;
    for (auto axis : axes)
    {
        prod *= shape.at(axis);
    }

    return prod;
}

template <typename T>
static shared_ptr<Node>
    multiply_by(element::Type type, size_t multiplier, shared_ptr<op::Constant> cnst)
{
    T sum_cnst = static_cast<T>(cnst->get_data_ptr<T>()[0] * multiplier);
    return op::Constant::create<T>(type, Shape{}, {sum_cnst});
}

template <typename T>
static shared_ptr<Node> pow_by(element::Type type, size_t multiplier, shared_ptr<op::Constant> cnst)
{
    T prod = static_cast<T>(1);
    T val = cnst->get_data_ptr<T>()[0];
    for (size_t i = 0; i < multiplier; i++)
    {
        prod *= val;
    }
    return op::Constant::create<T>(type, Shape{}, {prod});
}

static shared_ptr<Node> get_sum_constant(shared_ptr<op::Constant> cnst, size_t multiplier)
{
    if (cnst->get_element_type() == element::i32)
    {
        return multiply_by<int>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::i8)
    {
        return multiply_by<signed char>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f32)
    {
        return multiply_by<float>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f64)
    {
        return multiply_by<double>(cnst->get_element_type(), multiplier, cnst);
    }

    return nullptr;
}

static shared_ptr<Node> get_prod_constant(shared_ptr<op::Constant> cnst, size_t multiplier)
{
    if (cnst->get_element_type() == element::i32)
    {
        return pow_by<int>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::i8)
    {
        return pow_by<signed char>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f32)
    {
        return pow_by<float>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f64)
    {
        return pow_by<double>(cnst->get_element_type(), multiplier, cnst);
    }

    return nullptr;
}

//`simplify_reduction` optimizes the following case:
// sum(broadcast(scalar_constant), reduction_axes = ...) -> constant2 (or scalar constant)
// where constant2's values are equal to scalar_constant * shape_size(reduction_axes)
// product(broadcast(scalar_constant), reduction_axes = ...) -> constant2 (or scalar constant)
// where constant2's values are equal to scalar_constant ^ shape_size(reduction_axes)
template <typename T, shared_ptr<Node> (*F)(shared_ptr<op::Constant> cnst, size_t multiplier)>
static bool simplify_reduction(shared_ptr<Node> n)
{
    NGRAPH_DEBUG << "In simplify_reduction for " << n->get_name();
    if (n->get_output_partial_shape(0).is_dynamic())
    {
        NGRAPH_DEBUG << n << " has dynamic shape";
        return false;
    }
    auto reduction = static_pointer_cast<T>(n);

    auto broadcast = as_type_ptr<op::v0::Broadcast>(n->input_value(0).get_node_shared_ptr());
    if (!broadcast)
    {
        NGRAPH_DEBUG << n->get_name() << " isn't Broadcast";
        return false;
    }

    auto cnst = as_type_ptr<op::Constant>(broadcast->input_value(0).get_node_shared_ptr());
    if (!cnst || cnst->get_shape().size() > 0 /*not a scalar*/)
    {
        NGRAPH_DEBUG << broadcast->get_argument(0)->get_name() << " isn't a scalar constant";
        return false;
    }

    auto multiplier = reduction_shape_size(reduction->get_reduction_axes(), broadcast->get_shape());
    auto reduction_cnst = F(cnst, multiplier);

    // Unsupported type
    if (!reduction_cnst)
    {
        NGRAPH_DEBUG << "unsupported type";
        return false;
    }

    if (reduction->get_shape().size() > 0)
    {
        AxisSet axes{};
        for (size_t i = 0; i < reduction->get_shape().size(); i++)
        {
            axes.insert(i);
        }
        reduction_cnst =
            make_shared<op::v0::Broadcast>(reduction_cnst, reduction->get_shape(), axes);
    }

    replace_node(n, reduction_cnst);
    return true;
}

static bool replace_transpose_with_reshape(shared_ptr<Node> transpose)
{
    auto data = transpose->input_value(0);
    const auto input_shape = transpose->input(0).get_partial_shape();
    if (input_shape.rank().is_dynamic())
    {
        return false;
    }

    const auto input_shape_rank = input_shape.rank().get_length();

    auto order = as_type_ptr<opset3::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!order)
    {
        return false;
    }

    const auto order_value = order->cast_vector<int64_t>();

    // Check that transpose order without 1 dims has an ascending order
    int64_t last_dim(-1);
    for (size_t i = 0; i < input_shape_rank; ++i)
    {
        if (input_shape[order_value[i]].is_dynamic() || input_shape[order_value[i]] != 1)
        {
            if (order_value[i] < last_dim)
            {
                return false;
            }
            last_dim = order_value[i];
        }
    }

    // Transpose operation can be removed if original transpose order is sorted
    // or dimension that changes their places equal to 1
    using DimensionToPosition = struct
    {
        Dimension dim;
        size_t pos;
    };
    std::vector<DimensionToPosition> dims;
    for (size_t i = 0; i < input_shape_rank; ++i)
    {
        if (order_value[i] != i)
        {
            dims.push_back({input_shape[order_value[i]], i});
        }
    }

    // If number of dimensions != 1 to move equal to 0 we can remove this Transpose
    if (count_if(dims.begin(), dims.end(), [](const DimensionToPosition& item) {
            return !(item.dim.is_static() && item.dim.get_length() == 1);
        }) == 0)
    {
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
        }) < 2)
    {
        vector<int64_t> reshape_value(input_shape_rank, 0);
        for (const auto& item : dims)
        {
            reshape_value[item.pos] = item.dim.is_dynamic() ? -1 : item.dim.get_length();
        }
        reshape_dim =
            opset3::Constant::create(element::i64, Shape{reshape_value.size()}, reshape_value);
    }
    else
    {
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

static unordered_map<NodeTypeInfo, function<bool(shared_ptr<Node>)>> initialize_ops_to_simplifiers()
{
    return unordered_map<NodeTypeInfo, function<bool(shared_ptr<Node>)>>(
        {{op::v0::Add::type_info, simplify_add},
         {op::v0::Multiply::type_info, simplify_multiply},
         {opset3::Gather::type_info, simplify_gather},
         {op::v0::Concat::type_info, simplify_concat},
         {opset2::ShapeOf::type_info, simplify_gather_shapeof},
         {opset3::ShapeOf::type_info, simplify_gather_shapeof},
         {op::v0::Sum::type_info,
          function<bool(shared_ptr<Node>)>{simplify_reduction<op::v0::Sum, get_sum_constant>}},
         {op::v0::Product::type_info,
          function<bool(shared_ptr<Node>)>{simplify_reduction<op::v0::Product, get_prod_constant>}},
         {op::v0::Log::type_info, simplify_log},
         {opset3::Transpose::type_info, replace_transpose_with_reshape}});
}

static unordered_map<NodeTypeInfo, function<bool(shared_ptr<Node>)>> ops_to_simplifiers =
    initialize_ops_to_simplifiers();

bool pass::AlgebraicSimplification::run_on_function(shared_ptr<Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter())
        {
            continue;
        }

        auto eh = ops_to_simplifiers.find(n->get_type_info());
        if (eh != ops_to_simplifiers.end())
        {
            replaced = eh->second(n) || replaced;
        }
    }
    return replaced;
}
