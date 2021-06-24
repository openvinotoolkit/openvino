// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <numeric>
#include <set>

#include "transformations/common_optimizations/algebraic_simplification.hpp"
#include "itt.hpp"

#include <ngraph/log.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::AlgebraicSimplification, "AlgebraicSimplification", 0);

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
    auto new_shapeof = make_shared<opset3::ShapeOf>(gather->input_value(0), node->get_output_element_type(0));
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
        auto shapeof_indices = make_shared<opset3::ShapeOf>(gather->input_value(1), node->get_output_element_type(0));
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

#define ECHO(NAME) #NAME
#define STR(NAME) ECHO(NAME)
#define SIMPLE_MATCHER_PASS_DEFINITION(NAME, OP, FUNC) \
class NAME : public ngraph::pass::MatcherPass { \
public: \
NGRAPH_RTTI_DECLARATION; \
NAME() { \
    MATCHER_SCOPE(NAME);    \
    auto match_node = ngraph::pattern::wrap_type<OP>(); \
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) { \
        return FUNC(m.get_match_root()); \
    }; \
    auto m = std::make_shared<ngraph::pattern::Matcher>(match_node, matcher_name); \
    register_matcher(m, callback); \
}  \
}; \
NGRAPH_RTTI_DEFINITION(NAME, STR(NAME), 0);

SIMPLE_MATCHER_PASS_DEFINITION(EliminateGather, opset3::Gather, simplify_gather);
SIMPLE_MATCHER_PASS_DEFINITION(SimplifyShapeOf2Gather, opset2::ShapeOf, simplify_gather_shapeof);
SIMPLE_MATCHER_PASS_DEFINITION(SimplifyShapeOf3Gather, opset3::ShapeOf, simplify_gather_shapeof);

ngraph::pass::AlgebraicSimplification::AlgebraicSimplification() {
    add_matcher<EliminateGather>();
    add_matcher<SimplifyShapeOf2Gather>();
    add_matcher<SimplifyShapeOf3Gather>();
}
