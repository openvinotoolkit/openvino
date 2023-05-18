// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_gather.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

TSGatherForward::TSGatherForward() {
    MATCHER_SCOPE(TSGatherForward);

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({any_input(), wrap_type<ov::op::v0::Constant>()});
    auto gather_label =
        wrap_type<ov::op::v8::Gather>({transpose_label, any_input(), wrap_type<ov::op::v0::Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = as_type_ptr<ov::op::v1::Transpose>(pattern_to_output.at(transpose_label));
        auto main_node = as_type_ptr<ov::op::v8::Gather>(pattern_to_output.at(gather_label));
        if (transformation_callback(main_node) || !main_node) {
            return false;
        }

        auto transpose_order = as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
        auto gather_axis = as_type_ptr<ov::op::v0::Constant>(main_node->get_input_node_shared_ptr(2));
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

        const auto& order_val = transpose_order->cast_vector<size_t>();
        auto batch_dims = static_cast<size_t>(main_node->get_batch_dims());
        for (size_t i = 0; i < batch_dims; ++i) {
            // transpose changes the order of batch dims
            if (order_val[i] != i) {
                return false;
            }
        }

        size_t axis;
        if (axes[0] < 0) {
            auto data_rank = main_node->get_input_partial_shape(0).rank();
            if (data_rank.is_dynamic()) {
                return false;
            }
            axis = static_cast<size_t>(axes[0] + data_rank.get_length());
        } else {
            axis = static_cast<size_t>(axes[0]);
        }
        const auto& indices_rank_val = indices_rank.get_length();
        std::vector<size_t> new_transpose_order(order_val.size() + indices_rank_val - 1);
        for (size_t i = 0, j = 0; i < new_transpose_order.size(); ++i) {
            if (i > axis && i < (axis + indices_rank_val)) {
                new_transpose_order[i] = new_transpose_order[j - 1] + 1;
            } else if (order_val[i] > axis) {
                new_transpose_order[i] = order_val[j] + indices_rank_val - 1;
                j++;
            } else {
                new_transpose_order[i] = order_val[j];
                j++;
            }
        }

        auto new_order_const = ov::op::v0::Constant::create(transpose_order->get_element_type(),
                                                            {new_transpose_order.size()},
                                                            new_transpose_order);
        TransposeInputsInfo transpose_input_info = {transpose, new_order_const, 0};
        // deletes Transpose from 0 input
        auto success = sink_forward::UpdateInputTransposes(main_node, transpose_input_info, {0});
        if (!success) {
            return false;
        }
        auto new_axis =
            ov::op::v0::Constant::create(gather_axis->get_element_type(), gather_axis->get_shape(), {order_val[axis]});
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

    auto gather_label = wrap_type<ov::op::v8::Gather>({any_input(), any_input(), wrap_type<ov::op::v0::Constant>()},
                                                      CheckTransposeConsumers);
    auto transpose_label = wrap_type<ov::op::v1::Transpose>({gather_label, wrap_type<ov::op::v0::Constant>()},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = as_type_ptr<ov::op::v1::Transpose>(pattern_to_output.at(transpose_label));
        auto main_node = as_type_ptr<ov::op::v8::Gather>(pattern_to_output.at(gather_label));
        if (transformation_callback(main_node) || !main_node) {
            return false;
        }

        auto transpose_order = as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1));
        auto gather_axis = as_type_ptr<ov::op::v0::Constant>(main_node->get_input_node_shared_ptr(2));
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

        auto order_val = transpose_order->cast_vector<size_t>();
        auto batch_dims = static_cast<size_t>(main_node->get_batch_dims());
        for (size_t i = 0; i < batch_dims; ++i) {
            // transpose changes the order of batch dims
            if (order_val[i] != i) {
                return false;
            }
        }

        size_t axis;
        if (axes[0] < 0) {
            auto data_rank = main_node->get_input_partial_shape(0).rank();
            if (data_rank.is_dynamic()) {
                return false;
            }
            axis = static_cast<size_t>(axes[0] + data_rank.get_length());
        } else {
            axis = static_cast<size_t>(axes[0]);
        }
        auto out_pshape = main_node->get_output_partial_shape(0);
        bool optimization = out_pshape.is_static() && main_node->input_value(1).get_partial_shape().is_static();
        bool success = false;
        std::vector<size_t> axes_val;
        std::shared_ptr<ov::op::v0::Squeeze> squeeze;
        // In some cases shape of 2nd input to Gather op (indices) has `1` dims which can
        // prevent TransposeSinking in backward direction.
        // We can get around this case by wrapping Transpose op with Squeeze+Unsqueeze pair.
        /*
         * Data_input:shape(257, 8)       Indices_input: shape(1, 2)
                 │                               │
                 └────────────┐    ┌─────────────┘
                              ▼    ▼
                           Gather(axis = 0)
                                │
                                ▼
                         Gather output: shape(1,2,8)
                                │
                                │
                                ▼
                            Transpose
                                │
                                ▼
                         Transpose output: shape(1,8,2)
        */
        if (optimization) {
            squeeze = std::make_shared<ov::op::v0::Squeeze>(main_node->input_value(1));
            copy_runtime_info(main_node, squeeze);
            main_node->input(1).replace_source_output(squeeze);
            main_node->validate_and_infer_types();
            auto new_out_pshape = main_node->get_output_partial_shape(0);
            if (new_out_pshape.is_static()) {
                const auto shape = out_pshape.get_shape();
                const auto new_shape = new_out_pshape.get_shape();
                success = shape != new_shape;
                if (success) {
                    size_t j = 0;
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (shape[i] != new_shape[j] && shape[i] == 1) {
                            axes_val.push_back(i);
                            continue;
                        } else if (shape[i] != new_shape[j]) {
                            success = false;
                            break;
                        }
                        j++;
                    }
                    if (j != new_shape.size()) {
                        success = false;
                    }
                }
            }
            if (!success) {
                main_node->input(1).replace_source_output(squeeze->input_value(0));
            }
        }
        if (!axes_val.empty()) {
            order_val = GetOrderAfterReduction(axes_val, order_val);
        }

        const auto& indices_rank_val = static_cast<size_t>(main_node->get_input_partial_shape(1).rank().get_length());
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
                        if (success && squeeze) {
                            main_node->input(1).replace_source_output(squeeze->input_value(0));
                        }
                        return false;
                    }
                    prev_idx = i;
                }
            }
        }
        RemoveTransposeConsumers(main_node);
        if (success) {
            auto target_inputs = main_node->get_output_target_inputs(0);
            auto unsqueeze_axes = ov::op::v0::Constant::create(element::i32, {axes_val.size()}, axes_val);
            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(main_node, unsqueeze_axes);
            for (const auto& input : target_inputs) {
                input.replace_source_output(unsqueeze);
            }
            unsqueeze->output(0).add_names(main_node->output(0).get_names());
            main_node->output(0).set_names({});
            unsqueeze->set_friendly_name(main_node->get_friendly_name());
            main_node->set_friendly_name("");
            copy_runtime_info(main_node, {unsqueeze, unsqueeze_axes});
        }
        const auto reversed_transpose_order = ReverseTransposeOrder(order_val);
        const auto& transpose_const = ov::op::v0::Constant::create(transpose_order->get_element_type(),
                                                                   {new_transpose_order.size()},
                                                                   new_transpose_order);
        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node,
                                                                       transpose_const,
                                                                       /* input_indexes= */ {0})) {
            register_new_node(new_node);
        }
        auto new_axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, reversed_transpose_order[axis]);
        copy_runtime_info(gather_axis, new_axis);
        main_node->input(2).replace_source_output(new_axis);
        main_node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
