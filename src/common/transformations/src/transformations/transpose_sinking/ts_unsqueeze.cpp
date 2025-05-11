// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_unsqueeze.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

/**
 * @brief Checks that Reshape operation is equal to ov::op::v0::Unsqueeze:
 * Only 1 dims are inserted, all other dims must be the same.
 * Converts these 1 dims to axes format.
 * @arg reshape Reshape operation.
 * @arg reshape_to_shape 2nd input to Reshape op as a constant.
 * @arg result_axes contains axes which will be unsqueezed.
 */
bool shape_to_unsqueeze_axes(const std::shared_ptr<Node>& reshape,
                             const std::shared_ptr<ov::op::v0::Constant>& reshape_to_shape,
                             std::vector<size_t>& result_axes) {
    result_axes.clear();
    auto reduction_axes_values = reshape_to_shape->cast_vector<int64_t>();
    // supported the case if Reshape is equal to Unsqueeze
    const auto& new_shape = reduction_axes_values;
    const auto& input_pshape = reshape->get_input_partial_shape(0);
    // todo: support dynamic case
    if (input_pshape.is_dynamic()) {
        return false;
    }

    const auto input_shape = input_pshape.to_shape();
    if (new_shape.size() > input_shape.size()) {
        size_t j = 0;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (j < input_shape.size() && static_cast<int64_t>(input_shape[j]) == new_shape[i]) {
                j++;
            } else if (new_shape[i] != 1) {
                return false;
            } else {
                result_axes.push_back(i);
            }
        }
        if (j != input_shape.size()) {
            // not all input_shape values are in new_shape
            return false;
        }
    } else {
        // another reshape type, not Unsqueeze
        // todo: move this checks in the pattern
        return false;
    }
    return true;
}

/**
 * @brief Converts unsqueeze_axes to actual shape (2nd input) for Reshape operation
 * using the shape of the 1st input to Reshape.
 * @arg input_node 1st input to Reshape op.
 * @arg unsqueeze_axes In case of Reshape op is equal to ov::op::v0::Unsqueeze, these axes indicate the places where 1
 * dims have to be inserted.
 */
bool unsqueeze_axes_to_shape(const Output<Node>& input_node,
                             std::vector<size_t> unsqueeze_axes,
                             std::vector<size_t>& to_shape) {
    to_shape.clear();
    const auto& input_pshape = input_node.get_partial_shape();
    if (input_pshape.is_dynamic()) {
        return false;
    }
    const auto& input_shape = input_pshape.get_shape();
    to_shape.resize(input_shape.size() + unsqueeze_axes.size());
    std::sort(unsqueeze_axes.begin(), unsqueeze_axes.end());
    for (size_t i = 0, j = 0, k = 0; i < to_shape.size(); ++i) {
        if (j < unsqueeze_axes.size() && i == unsqueeze_axes[j]) {
            to_shape[i] = 1;
            j++;
        } else if (k < input_shape.size()) {
            to_shape[i] = input_shape[k];
            k++;
        }
    }
    return true;
}

bool AreInputOutputShapesEqual(const std::shared_ptr<ov::op::v1::Reshape>& reshape) {
    const auto input_shape = reshape->get_input_partial_shape(0);
    const auto output_shape = reshape->get_output_partial_shape(0);

    if (input_shape.is_dynamic() || output_shape.is_dynamic()) {
        return false;
    }
    return input_shape == output_shape;
}

bool HasSpecialOne(const std::shared_ptr<ov::op::v0::Constant>& reshape_const) {
    auto const_value = reshape_const->cast_vector<int64_t>();
    return std::find(const_value.begin(), const_value.end(), -1) != const_value.end();
}

}  // namespace

TSUnsqueezeForward::TSUnsqueezeForward() {
    MATCHER_SCOPE(TSUnsqueezeForward);

    create_pattern<ov::op::v0::Unsqueeze, ov::op::v1::Reshape>({0});

    auto sinking_transformation = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<Node>& main_node,
                                                            const TransposeInputsInfo& transpose_info) -> bool {
        auto unsqueeze_axes = as_type_ptr<ov::op::v0::Constant>(main_node->get_input_node_shared_ptr(1));
        if (!unsqueeze_axes) {
            return false;
        }
        auto ts_order_values = transpose_info.transpose_const->cast_vector<size_t>();

        // if main_node does nothing, just swap them
        auto reshape = as_type_ptr<ov::op::v1::Reshape>(main_node);
        if (reshape && AreInputOutputShapesEqual(reshape) && !HasSpecialOne(unsqueeze_axes)) {
            TransposeInputsInfo transpose_input_info = {transpose_info.transpose, transpose_info.transpose_const, 0};
            // remove input Transpose
            auto success = sink_forward::UpdateInputTransposes(main_node, transpose_input_info, {0});
            if (!success) {
                return false;
            }

            const auto reshape_order = ov::pass::transpose_sinking::utils::ReverseTransposeOrder(ts_order_values);
            // transpose reshape const with Gather operation
            auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
            auto gather =
                ov::pass::transpose_sinking::utils::ChangeValuesOrder(reshape->input_value(1), reshape_order, axis);
            main_node->input(1).replace_source_output(gather);

            default_outputs_update(main_node, transpose_input_info);
            return true;
        }

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
            auto success = shape_to_unsqueeze_axes(main_node, unsqueeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            auto rank = main_node->get_output_partial_shape(0).rank();
            non_negative_axes =
                ov::util::try_get_normalized_axis_vector(unsqueeze_axes->get_tensor_view(), rank, *main_node);
        }

        ts_order_values = GetOrderBeforeReduction(non_negative_axes, ts_order_values);
        auto new_transpose_order = ov::op::v0::Constant::create(transpose_info.transpose_const->get_element_type(),
                                                                {ts_order_values.size()},
                                                                ts_order_values);

        if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
            std::vector<size_t> new_values;
            auto success =
                unsqueeze_axes_to_shape(transpose_info.transpose->input_value(0), non_negative_axes, new_values);
            if (!success) {
                return false;
            }
            auto new_const =
                ov::op::v0::Constant::create(unsqueeze_axes->get_element_type(), {new_values.size()}, new_values);
            main_node->input(1).replace_source_output(new_const);
            copy_runtime_info(unsqueeze_axes, new_const);
        }

        TransposeInputsInfo transpose_input_info = {transpose_info.transpose, new_transpose_order, 0};
        // deletes Transpose from 0 input
        auto success = sink_forward::UpdateInputTransposes(main_node, transpose_input_info, {0});
        if (!success) {
            return false;
        }

        default_outputs_update(main_node, transpose_input_info);
        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}

TSUnsqueezeBackward::TSUnsqueezeBackward() {
    MATCHER_SCOPE(TSUnsqueezeBackward);

    auto unsqueeze_label =
        wrap_type<ov::op::v0::Unsqueeze, ov::op::v1::Reshape>({any_input(), wrap_type<ov::op::v0::Constant>()},
                                                              CheckTransposeConsumers);
    auto transpose_label = wrap_type<ov::op::v1::Transpose>({unsqueeze_label, wrap_type<ov::op::v0::Constant>()},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto main_node = pattern_to_output.at(unsqueeze_label);
        if (transformation_callback(main_node)) {
            return false;
        }

        auto transpose_order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
        auto unsqueeze_axes = ov::as_type_ptr<ov::op::v0::Constant>(main_node->get_input_node_shared_ptr(1));
        if (!transpose_order || !unsqueeze_axes)
            return false;

        auto transpose_order_values = transpose_order->cast_vector<size_t>();

        // if main_node does nothing, just swap them
        auto reshape = as_type_ptr<ov::op::v1::Reshape>(main_node);
        if (reshape && AreInputOutputShapesEqual(reshape) && !HasSpecialOne(unsqueeze_axes)) {
            // insert Transpose before main_node on #0 input
            for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, transpose_order, {0})) {
                register_new_node(new_node);
            }
            // transpose reshape const with Gather operation
            auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
            auto gather = ov::pass::transpose_sinking::utils::ChangeValuesOrder(reshape->input_value(1),
                                                                                transpose_order_values,
                                                                                axis);
            main_node->input(1).replace_source_output(gather);

            main_node->validate_and_infer_types();
            RemoveTransposeConsumers(main_node);
            return true;
        }

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
            auto success = shape_to_unsqueeze_axes(main_node, unsqueeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            const auto& axes = unsqueeze_axes->cast_vector<int64_t>();
            if (std::all_of(axes.begin(), axes.end(), [](int64_t axis) {
                    return axis >= 0;
                })) {
                non_negative_axes = std::vector<size_t>(axes.begin(), axes.end());
            } else {
                auto rank = main_node->get_output_partial_shape(0).rank();
                if (rank.is_dynamic()) {
                    return false;
                }
                non_negative_axes =
                    util::try_get_normalized_axis_vector(unsqueeze_axes->get_tensor_view(), rank, *main_node);
            }
        }

        auto old_transpose_order_values = transpose_order_values;
        std::vector<size_t> new_values;

        if (non_negative_axes.size() == transpose_order_values.size()) {
            // input is a scalar, we main_node all dims
            // it's enough to eliminate such Transpose
            transpose->output(0).replace(main_node);
            return true;
        }

        for (const auto& axis : non_negative_axes) {
            auto it = std::find(old_transpose_order_values.begin(), old_transpose_order_values.end(), axis);
            if (it != old_transpose_order_values.end()) {
                new_values.push_back(it - old_transpose_order_values.begin());
            }
        }

        transpose_order_values = GetOrderAfterReduction(new_values, transpose_order_values);
        auto new_transpose_order = std::make_shared<ov::op::v0::Constant>(transpose_order->get_element_type(),
                                                                          Shape{transpose_order_values.size()},
                                                                          transpose_order_values);

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, new_transpose_order, {0})) {
            register_new_node(new_node);
        }
        if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
            std::vector<size_t> to_shape;
            auto success = unsqueeze_axes_to_shape(main_node->input_value(0), new_values, to_shape);
            if (!success) {
                return false;
            }
            new_values = to_shape;
        }
        auto new_const =
            ov::op::v0::Constant::create(unsqueeze_axes->get_element_type(), {new_values.size()}, new_values);
        main_node->input(1).replace_source_output(new_const);

        main_node->validate_and_infer_types();
        RemoveTransposeConsumers(main_node);
        copy_runtime_info(unsqueeze_axes, new_const);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
