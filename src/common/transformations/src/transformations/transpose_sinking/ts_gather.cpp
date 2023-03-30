// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_gather.hpp"

#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

TSGatherForward::TSGatherForward() {
    MATCHER_SCOPE(TSGatherForward);

    auto transpose_label = wrap_type<Transpose>({any_input(), wrap_type<Constant>()});
    auto gather_label = wrap_type<Gather>({transpose_label, any_input(), wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = as_type_ptr<Transpose>(pattern_to_output.at(transpose_label));
        auto main_node = as_type_ptr<Gather>(pattern_to_output.at(gather_label));
        if (transformation_callback(main_node) || !main_node) {
            return false;
        }

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto gather_axis = as_type_ptr<Constant>(main_node->get_input_node_shared_ptr(2));
        if (!transpose || !transpose_order || !gather_axis) {
            return false;
        }

        const auto& axes = gather_axis->cast_vector<int64_t>();
        if (axes.size() != 1) {
            return false;
        }

        const auto& indices_rank = main_node->get_input_partial_shape(1).rank();
        if (indices_rank.is_dynamic()) {
            return false;
        }

        const auto& order_val = transpose_order->cast_vector<int64_t>();
        auto batch_dims = main_node->get_batch_dims();
        for (int64_t i = 0; i < batch_dims; ++i) {
            // transpose changes the order of batch dims
            if (order_val[i] != i) {
                return false;
            }
        }

        const auto& axis = axes[0];
        const auto& indices_rank_val = indices_rank.get_length();
        std::vector<size_t> new_transpose_order(order_val.size() + indices_rank_val - 1);
        for (size_t i = 0, j = 0; i < new_transpose_order.size(); ++i) {
            if (i > axis && i < (axis + indices_rank_val)) {
                new_transpose_order[i] = new_transpose_order[j-1] + 1;
            } else if (order_val[i] > axis) {
                new_transpose_order[i] = order_val[j] + indices_rank_val - 1;
                j++;
            } else {
                new_transpose_order[i] = order_val[j];
                j++;
            }
        }

        auto new_order_const = Constant::create(transpose_order->get_element_type(), {new_transpose_order.size()}, new_transpose_order);
        TransposeInputsInfo transpose_input_info = {transpose, new_order_const, 0};
        // deletes Transpose from 0 input
        auto success = sink_forward::UpdateInputTransposes(main_node, transpose_input_info, {0});
        if (!success) {
            return false;
        }
        auto new_axis = Constant::create(gather_axis->get_element_type(), gather_axis->get_shape(), {order_val[axis]});
        main_node->input(2).replace_source_output(new_axis);
        copy_runtime_info(gather_axis, new_axis);
        main_node->validate_and_infer_types();
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
            UpdateForwardSinkingAbility(new_node);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSGatherBackward::TSGatherBackward() {
    MATCHER_SCOPE(TSGatherBackward);

    auto gather_label = wrap_type<Gather>({any_input(), any_input(), wrap_type<Constant>()}, HasSameOutputTransposeNodes);
    auto transpose_label =
        wrap_type<Transpose>({gather_label, wrap_type<Constant>()}, [](const Output<Node>& output) -> bool {
            return has_static_rank()(output) && is_sinking_node(output);
        });

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = as_type_ptr<Transpose>(pattern_to_output.at(transpose_label));
        auto main_node = as_type_ptr<Gather>(pattern_to_output.at(gather_label));
        if (transformation_callback(main_node) || !main_node) {
            return false;
        }

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto gather_axis = as_type_ptr<Constant>(main_node->get_input_node_shared_ptr(2));
        if (!transpose || !transpose_order || !gather_axis) {
            return false;
        }

        const auto& axes = gather_axis->cast_vector<int64_t>();
        if (axes.size() != 1) {
            return false;
        }

        const auto& indices_rank = main_node->get_input_partial_shape(1).rank();
        if (indices_rank.is_dynamic()) {
            return false;
        }

        const auto& order_val = transpose_order->get_axis_vector_val();
        auto batch_dims = main_node->get_batch_dims();
        for (int64_t i = 0; i < batch_dims; ++i) {
            // transpose changes the order of batch dims
            if (order_val[i] != i) {
                return false;
            }
        }

        RemoveSingleOutputConsumers(main_node);
        SwapNames(main_node, transpose);

        const auto& axis = axes[0];
        const auto& indices_rank_val = indices_rank.get_length();
        std::vector<size_t> new_transpose_order(order_val.size() - indices_rank_val + 1);
        for (size_t i = 0, j = 0; i < order_val.size(); ++j) {
            if (order_val[i] < axis) {
                new_transpose_order[j] = order_val[i];
                ++i;
            } else if (order_val[i] > axis) {
                new_transpose_order[j] = order_val[i] - indices_rank_val + 1;
                ++i;
            } else {
                // the next `indices_rank_val` values have to be in ascending order
                // these values will be replaced with a single axis
                new_transpose_order[j] = order_val[i];
                size_t prev_idx = i;
                for (size_t k = 0; i < order_val.size() && k < indices_rank_val; ++i, ++k) {
                    if (order_val[i] != order_val[prev_idx]) {
                        return false;
                    }
                    prev_idx = i;
                }
            }
        }
        const auto reversed_transpose_order = ReverseTransposeOrder(order_val);
        const auto& transpose_const = Constant::create(transpose_order->get_element_type(), {new_transpose_order.size()}, new_transpose_order);
        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node,
                                                                       transpose_const,
                /* input_indexes= */ {0})) {
            register_new_node(new_node);
        }
        auto new_axis = std::make_shared<Constant>(element::i32, Shape{1}, reversed_transpose_order[axis]);
        copy_runtime_info(gather_axis, new_axis);
        main_node->input(2).replace_source_output(new_axis);
        main_node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
