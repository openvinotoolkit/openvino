// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_squeeze.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

/**
 * @brief Checks that Reshape operation is equal to ov::op::v0::Squeeze:
 * Only 1 dims are deleted, all other dims must be the same.
 * Converts these 1 dims to axes format.
 * @arg reshape Reshape operation.
 * @arg reshape_to_shape 2nd input to Reshape op as a constant.
 * @arg result_axes Contains axes which will be squeezed.
 */
bool shape_to_squeeze_axes(const std::shared_ptr<Node>& reshape,
                           const std::shared_ptr<ov::op::v0::Constant>& reshape_to_shape,
                           std::vector<size_t>& result_axes) {
    result_axes.clear();
    auto reduction_axes_values = reshape_to_shape->cast_vector<int64_t>();
    // supported the case if Reshape is equal to Squeeze
    const auto& new_shape = reduction_axes_values;
    const auto& input_pshape = reshape->get_input_partial_shape(0);
    // todo: support dynamic case
    if (input_pshape.is_dynamic()) {
        return false;
    }

    const auto input_shape = input_pshape.to_shape();
    if (new_shape.size() < input_shape.size()) {
        size_t j = 0;
        for (size_t i = 0; i < input_shape.size(); i++) {
            const auto input_dim = static_cast<int64_t>(input_shape[i]);
            if (j < new_shape.size() && new_shape[j] == input_dim) {
                j++;
            } else if (input_dim != 1) {
                return false;
            } else {
                result_axes.push_back(i);
            }
        }
        if (j != new_shape.size()) {
            // not all new_shape values are in input_shape
            return false;
        }
    } else {
        // another reshape type, not Squeeze
        // todo: move this checks in the pattern
        return false;
    }
    return true;
}

/**
 * @brief Converts squeezed_axes to actual shape (2nd input) for Reshape operation
 * using the shape of the 1st input to Reshape.
 * @arg input_node 1st input to Reshape op.
 * @arg squeeze_axes In case of Reshape op is equal to squeeze, these axes indicate the places where 1 dims have
 * to be deleted.
 */
bool squeeze_axes_to_shape(const Output<Node>& input_node,
                           std::vector<size_t> squeeze_axes,
                           std::vector<size_t>& to_shape) {
    to_shape.clear();
    std::sort(squeeze_axes.begin(), squeeze_axes.end());
    const auto& input_pshape = input_node.get_partial_shape();
    if (input_pshape.is_dynamic()) {
        return false;
    }
    const auto& input_shape = input_pshape.get_shape();
    for (size_t i = 0, j = 0; i < input_shape.size(); ++i) {
        if (j < squeeze_axes.size() && i == squeeze_axes[j]) {
            ++j;
            continue;
        }
        to_shape.push_back(input_shape[i]);
    }
    return true;
}

}  // namespace

TSSqueezeForward::TSSqueezeForward() {
    MATCHER_SCOPE(TSSqueezeForward);

    create_pattern<ov::op::v0::Squeeze, ov::op::v1::Reshape>({0});

    auto sinking_transformation = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<Node>& main_node,
                                                            const TransposeInputsInfo& transpose_info) -> bool {
        std::vector<size_t> non_negative_axes;
        std::shared_ptr<ov::op::v0::Constant> squeeze_axes;
        if (main_node->get_input_size() > 1) {
            squeeze_axes = as_type_ptr<ov::op::v0::Constant>(main_node->get_input_node_shared_ptr(1));
            if (!squeeze_axes) {
                return false;
            }
            if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
                auto success = shape_to_squeeze_axes(main_node, squeeze_axes, non_negative_axes);
                if (!success) {
                    return false;
                }
            } else {
                auto rank = main_node->get_input_partial_shape(0).rank();
                non_negative_axes =
                    util::try_get_normalized_axis_vector(squeeze_axes->get_tensor_view(), rank, *main_node);
            }
        }

        // if 2nd input to main_node is empty then all '1' dims will be deleted.
        if (non_negative_axes.empty()) {
            auto input_pshape = transpose_info.transpose->output(0).get_partial_shape();
            if (input_pshape.is_dynamic()) {
                return false;
            }
            for (size_t i = 0; i < input_pshape.size(); ++i) {
                if (input_pshape[i].get_length() == 1) {
                    non_negative_axes.push_back(i);
                }
            }
        }

        auto transpose_order_values = transpose_info.transpose_const->cast_vector<size_t>();
        std::vector<size_t> new_values;
        new_values.reserve(non_negative_axes.size());
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(transpose_order_values[axis]);
        }

        transpose_order_values = GetOrderAfterReduction(non_negative_axes, transpose_order_values);
        auto new_transpose_order = ov::op::v0::Constant::create(transpose_info.transpose_const->get_element_type(),
                                                                {transpose_order_values.size()},
                                                                transpose_order_values);

        if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
            std::vector<size_t> to_shape;
            auto success = squeeze_axes_to_shape(transpose_info.transpose->input_value(0), new_values, to_shape);
            if (!success) {
                return false;
            }
            new_values = to_shape;
        }

        if (squeeze_axes) {
            auto new_const =
                ov::op::v0::Constant::create(squeeze_axes->get_element_type(), {new_values.size()}, new_values);
            main_node->input(1).replace_source_output(new_const);
            copy_runtime_info(squeeze_axes, new_const);
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

TSSqueezeBackward::TSSqueezeBackward() {
    MATCHER_SCOPE(TSSqueezeBackward);
    auto squeeze_with_1_input = wrap_type<ov::op::v0::Squeeze>({any_input()}, CheckTransposeConsumers);
    auto squeeze_label =
        wrap_type<ov::op::v0::Squeeze, ov::op::v1::Reshape>({any_input(), wrap_type<ov::op::v0::Constant>()},
                                                            CheckTransposeConsumers);
    auto pattern = std::make_shared<pattern::op::Or>(OutputVector{squeeze_with_1_input, squeeze_label});
    auto transpose_label = wrap_type<ov::op::v1::Transpose>({pattern, wrap_type<ov::op::v0::Constant>()},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        std::shared_ptr<Node> main_node;
        if (pattern_to_output.count(squeeze_label)) {
            main_node = pattern_to_output.at(squeeze_label);
        } else {
            main_node = pattern_to_output.at(squeeze_with_1_input);
        }

        if (transformation_callback(main_node)) {
            return false;
        }

        auto transpose_order = as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));

        if (!transpose_order) {
            return false;
        }

        std::vector<size_t> non_negative_axes;
        std::shared_ptr<ov::op::v0::Constant> squeeze_axes;
        if (main_node->get_input_size() > 1) {
            squeeze_axes = as_type_ptr<ov::op::v0::Constant>(main_node->get_input_node_shared_ptr(1));
            if (!squeeze_axes) {
                return false;
            }
            if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
                auto success = shape_to_squeeze_axes(main_node, squeeze_axes, non_negative_axes);
                if (!success) {
                    return false;
                }
            } else {
                auto rank = main_node->get_input_partial_shape(0).rank();
                non_negative_axes =
                    util::try_get_normalized_axis_vector(squeeze_axes->get_tensor_view(), rank, *main_node);
            }
        }

        bool squeeze_all_dims = false;
        if (non_negative_axes.empty()) {
            auto input_pshape = main_node->input_value(0).get_partial_shape();
            if (input_pshape.is_dynamic()) {
                return false;
            }
            for (size_t i = 0; i < input_pshape.size(); ++i) {
                if (input_pshape[i] == 1) {
                    non_negative_axes.push_back(i);
                }
            }
            squeeze_all_dims = true;
        }

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        transpose_order_values = GetOrderBeforeReduction(non_negative_axes, transpose_order_values);
        auto reversed_order_values = ReverseTransposeOrder(transpose_order_values);

        std::vector<size_t> new_values;
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(reversed_order_values[axis]);
        }

        auto new_transpose_order = ov::op::v0::Constant::create(transpose_order->get_element_type(),
                                                                {transpose_order_values.size()},
                                                                transpose_order_values);
        auto new_transpose = transpose->clone_with_new_inputs({main_node->input_value(0), new_transpose_order});
        if (as_type_ptr<ov::op::v1::Reshape>(main_node)) {
            std::vector<size_t> to_shape;
            auto success = squeeze_axes_to_shape(new_transpose->output(0), new_values, to_shape);
            if (!success) {
                return false;
            }
            new_values = to_shape;
        }

        std::shared_ptr<Node> new_squeeze;
        if (!squeeze_all_dims) {
            auto new_const =
                ov::op::v0::Constant::create(squeeze_axes->get_element_type(), {new_values.size()}, new_values);
            main_node->input(1).replace_source_output(new_const);
            copy_runtime_info(squeeze_axes, new_const);
        }

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, new_transpose_order, {0})) {
            register_new_node(new_node);
        }
        main_node->validate_and_infer_types();
        RemoveTransposeConsumers(main_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
