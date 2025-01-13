// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pull_through_reduce.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"
#include "openvino/op/util/reduction_base.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "sequence_generator.hpp"
#include "transformations/utils/utils.hpp"

namespace {
// Adjust axes of Unsqueeze/Reduce ops after Unsqueeze pulling
// For example if we have:
//          input(shape={5,10,15})
//                 |
//          Unsqueeze(axes=[0,3]) -> output_shape = {1,5,10,1,15}
//                 |
// ReduceOp(axes=[2,4], keep_dims=false) -> output_shape = {1,5,1}
// after pulling it will be:
//          input(shape={5,10,15})
//                 |
// ReduceOp(axes=[1,2], keep_dims=false) -> output_shape = {5}
//                 |
//          Unsqueeze(axes=[0,2]) -> output_shape = {1,5,1}
const std::vector<int64_t> adjust_axes(const std::vector<int64_t>& axes_to_align,
                                       const std::vector<int64_t>& offset_axes) {
    auto number_of_axes_less_than = [&offset_axes](const int64_t current_axis) {
        return std::count_if(std::begin(offset_axes),
                             std::end(offset_axes),
                             [&current_axis](const int64_t excluded_axis) {
                                 return excluded_axis < current_axis;
                             });
    };
    std::vector<int64_t> result;
    for (const auto& axis : axes_to_align) {
        result.push_back(axis - number_of_axes_less_than(axis));
    }
    return result;
}

// Try to represent given Reshape node via Unsqueeze and calculate axes of such Unsqueeze
// - Reshape(input_shape={5,10,15}, target_shape={5,10,15,1}), 3 axis returned
// - Reshape(input_shape={5,10,15}, target_shape={1,5,10,15}), 0 axis returned
// - Reshape(input_shape={5,10,15}, target_shape={1,5,10,15,1}), 0 and 3 axes returned
// - Reshape(input_shape={5,10,15}, target_shape={5,10,1,15}), 2 axis is returned
std::vector<int64_t> try_get_unsqueeze_axes_from_reshape(const ov::Shape& target_shape, const ov::Shape& input_shape) {
    std::vector<int64_t> result;
    if (target_shape.size() <= input_shape.size()) {
        return result;
    }
    if (input_shape.size() == 0) {  // scalar case - can be reshaped only to [1,..,1] shape
        result.resize(target_shape.size(), 0);
        std::iota(std::begin(result), std::end(result), 0);
        return result;
    }
    size_t cur_input_shape_elem_idx = 0;
    auto cur_input_shape_elem = input_shape[cur_input_shape_elem_idx];
    size_t target_shape_idx = 0;
    for (; target_shape_idx < target_shape.size(); ++target_shape_idx) {
        if (cur_input_shape_elem == target_shape[target_shape_idx] &&
            cur_input_shape_elem_idx + 1 < input_shape.size()) {
            ++cur_input_shape_elem_idx;
            cur_input_shape_elem = input_shape[cur_input_shape_elem_idx];
        } else if (target_shape[target_shape_idx] == 1 &&
                   (target_shape_idx >= input_shape.size() + result.size() || cur_input_shape_elem != 1)) {
            result.push_back(target_shape_idx);
        }
    }
    if (cur_input_shape_elem_idx == input_shape.size() - 1 && target_shape_idx == target_shape.size()) {
        return result;
    } else {
        return {};
    }
    return result;
}

// Update given reshape_input_shape by inserting "1" dimension on the postion represented by axes_to_insert
std::shared_ptr<ov::op::v0::Constant> update_reshape_target_shape(const ov::Shape& reshape_input_shape,
                                                                  const std::vector<int64_t>& axes_to_insert) {
    auto result = std::vector<int64_t>(std::begin(reshape_input_shape), std::end(reshape_input_shape));
    for (const auto& axis : axes_to_insert) {
        result.insert(std::next(std::begin(result), axis), 1);
    }
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{result.size()}, result);
}

// Return true if given inputs have some common elements, otherwise return false.
bool have_same_axes(const std::vector<int64_t>& unsqueeze_axes, const std::vector<int64_t>& reduce_op_axes) {
    return std::find_first_of(std::begin(unsqueeze_axes),
                              std::end(unsqueeze_axes),
                              std::begin(reduce_op_axes),
                              std::end(reduce_op_axes)) != std::end(unsqueeze_axes);
}
}  // namespace

ov::pass::PullUnsqueezeThroughReduce::PullUnsqueezeThroughReduce() {
    MATCHER_SCOPE(PullUnsqueezeThroughReduce);

    const auto input = pattern::any_input(pattern::has_static_rank());
    const auto unsqueeze_axes = pattern::wrap_type<ov::op::v0::Constant>();
    const auto unsqueeze =
        pattern::wrap_type<ov::op::v0::Unsqueeze>({input, unsqueeze_axes}, pattern::consumers_count(1));
    const auto reduce_axes = pattern::wrap_type<ov::op::v0::Constant>();
    const auto reduce = pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims>(
        {unsqueeze, reduce_axes});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        const auto input_node = pattern_map.at(input);
        const auto reduce_node = ov::as_type_ptr<op::util::ReductionBase>(pattern_map.at(reduce).get_node_shared_ptr());
        const auto unsqueeze_node = pattern_map.at(unsqueeze).get_node_shared_ptr();
        auto unsqueeze_axes_input =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(unsqueeze_axes).get_node_shared_ptr());
        auto reduce_axes_input =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(reduce_axes).get_node_shared_ptr());

        if (!unsqueeze_axes_input || !reduce_axes_input || !reduce_node) {
            return false;
        }

        if (unsqueeze_node->get_output_partial_shape(0).rank().is_dynamic()) {
            return false;
        }

        auto unsqueeze_axes_val = unsqueeze_axes_input->cast_vector<int64_t>();
        ov::util::try_normalize_axes(unsqueeze_axes_val,
                                     unsqueeze_node->get_output_partial_shape(0).rank(),
                                     *unsqueeze_node);
        const auto reduce_axes_val = reduce_node->get_reduction_axes().to_vector();

        if (have_same_axes(unsqueeze_axes_val, reduce_axes_val)) {
            return false;
        }

        const bool keep_dims = reduce_node->get_keep_dims();

        if (!keep_dims) {
            const auto unsqueeze_adjusted_axes = adjust_axes(unsqueeze_axes_val, reduce_axes_val);
            if (unsqueeze_adjusted_axes != unsqueeze_axes_val) {
                unsqueeze_axes_input = ov::op::v0::Constant::create(unsqueeze_axes_input->get_element_type(),
                                                                    unsqueeze_axes_input->get_shape(),
                                                                    unsqueeze_adjusted_axes);
            }
        }

        const auto reduce_adjusted_axes = adjust_axes(reduce_axes_val, unsqueeze_axes_val);
        if (reduce_adjusted_axes != reduce_axes_val) {
            reduce_axes_input = ov::op::v0::Constant::create(reduce_axes_input->get_element_type(),
                                                             reduce_axes_input->get_shape(),
                                                             reduce_adjusted_axes);
        }

        const auto new_reduce_node = reduce_node->clone_with_new_inputs({input_node, reduce_axes_input});
        new_reduce_node->set_friendly_name(unsqueeze_node->get_friendly_name());
        const auto new_unsqueeze_node = unsqueeze_node->clone_with_new_inputs({new_reduce_node, unsqueeze_axes_input});
        new_unsqueeze_node->set_friendly_name(reduce_node->get_friendly_name());

        copy_runtime_info({reduce_node, unsqueeze_node}, {new_reduce_node, new_unsqueeze_node});
        replace_node(m.get_match_root(), new_unsqueeze_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PullReshapeThroughReduce::PullReshapeThroughReduce() {
    MATCHER_SCOPE(PullReshapeThroughReduce);

    const auto input = pattern::any_input(pattern::has_static_shape());
    const auto reshape_target_shape = pattern::wrap_type<ov::op::v0::Constant>();
    const auto reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({input, reshape_target_shape}, pattern::consumers_count(1));
    const auto reduce_axes = pattern::wrap_type<ov::op::v0::Constant>();
    const auto reduce = pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims>(
        {reshape, reduce_axes});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        const auto input_node = pattern_map.at(input);
        const auto reduce_node = ov::as_type_ptr<op::util::ReductionBase>(pattern_map.at(reduce).get_node_shared_ptr());
        if (!reduce_node) {
            return false;
        }
        const auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        if (reshape_node->get_output_partial_shape(0).is_dynamic()) {
            return false;
        }
        const auto unsqueeze_axes =
            try_get_unsqueeze_axes_from_reshape(reshape_node->get_shape(), input_node.get_shape());
        if (unsqueeze_axes.empty()) {
            return false;
        }

        const auto reduce_axes_val = reduce_node->get_reduction_axes().to_vector();

        if (have_same_axes(unsqueeze_axes, reduce_axes_val)) {
            return false;
        }

        const auto unsqueeze_adjusted_axes = adjust_axes(unsqueeze_axes, reduce_axes_val);
        const auto reduce_adjusted_axes = adjust_axes(reduce_axes_val, unsqueeze_axes);

        auto reduce_axes_input =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(reduce_axes).get_node_shared_ptr());

        if (!reduce_axes_input) {
            return false;
        }

        if (reduce_adjusted_axes != reduce_axes_val) {
            reduce_axes_input = ov::op::v0::Constant::create(reduce_axes_input->get_element_type(),
                                                             reduce_axes_input->get_shape(),
                                                             reduce_adjusted_axes);
        }

        const auto new_reduce_node = reduce_node->clone_with_new_inputs({input_node, reduce_axes_input});
        new_reduce_node->set_friendly_name(reshape_node->get_friendly_name());
        const auto new_reshape_node = reshape_node->clone_with_new_inputs(
            {new_reduce_node, update_reshape_target_shape(new_reduce_node->get_shape(), unsqueeze_adjusted_axes)});
        new_reshape_node->set_friendly_name(reduce_node->get_friendly_name());

        copy_runtime_info({reduce_node, reshape_node}, {new_reduce_node, new_reshape_node});
        replace_node(m.get_match_root(), new_reshape_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}
