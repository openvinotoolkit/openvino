// Copyright (C) 2018-2025 Intel Corporation
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

    create_pattern<ov::op::v8::Gather>({0});

    auto sinking_transformation = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<Node>& main_node,
                                                            const TransposeInputsInfo& transpose_info) -> bool {
        auto gather = as_type_ptr<ov::op::v8::Gather>(main_node);
        if (!gather) {
            return false;
        }

        auto transpose_order = transpose_info.transpose_const;
        auto gather_axis = as_type_ptr<ov::op::v0::Constant>(main_node->get_input_node_shared_ptr(2));
        if (!gather_axis) {
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
        auto batch_dims = static_cast<size_t>(gather->get_batch_dims());
        for (size_t i = 0; i < batch_dims; ++i) {
            // transpose changes the order of batch dims
            if (order_val[i] != i) {
                return false;
            }
        }

        size_t order_axis;
        if (axes[0] < 0) {
            auto data_rank = main_node->get_input_partial_shape(0).rank();
            if (data_rank.is_dynamic()) {
                return false;
            }
            order_axis = static_cast<size_t>(axes[0] + data_rank.get_length());
        } else {
            order_axis = static_cast<size_t>(axes[0]);
        }
        const size_t axis = order_val[order_axis];
        /*
            https://docs.openvino.ai/2023.0/openvino_docs_ops_movement_Gather_8.html
            The Gather output shape has the same shape as the input,
            with the indexed-axis replaced by the shape of the indices
            Gather input shape | Gather indexes shape | axis | Gather output shape
                {1, 2, 3}      |        {}            |   1  |      {1, 3}
                {1, 2, 3}      |        {7}           |   1  |      {1, 7, 3}
                {1, 2, 3}      |        {7,5}         |   1  |      {1, 7, 5, 3}

            New transpose order length equals to output Gather shape size.
            As gather modifies input shape within axis dimension, our transpose order
            will be modified with axis dimension.
            New transpose order values:
                - values before axis will be original
                - values in [axis, axis + indexes_ranks_size - 1] will be original + [0 1 ...]
                  if indexes_ranks_size == 0, there will be no such items
                - values after axis will be original + indexes_rank_size - 1
                  (as one dim[axis] will be substituted with new indexes_rank_size dimesions)
                  if indexes_ranks_size == 0, values will be original - 1
        */
        const auto& indices_rank_val = indices_rank.get_length();
        std::vector<size_t> new_transpose_order(order_val.size() + indices_rank_val - 1);
        const int n_axis_dims = static_cast<int>(indices_rank_val) - 1;
        /*
            i - new_transpose_order index
            j - order_val index
            k - substituted dims by Gather index
            - There might be a situation when output Gather shape has one dimension
                less than input shape. In a such case n_axis_dims < 0 and we should
                skip order_val[axis] and all the next order_val[j] will be reduced.
            - On the other hand in a case with multidimentional index Gather output
                shape has more dimensions than input shape. We need to add this
                dimensions into the transpose order and increase all next order_val[j]
        */
        for (size_t i = 0, j = 0, k = 0; i < new_transpose_order.size(); ++i) {
            if (order_val[j] == axis && static_cast<int>(k) > n_axis_dims) {
                /*
                    We added all new dimensions into the order.
                    We should go to the next order_val value.
                */
                ++j;
            }
            if (order_val[j] < axis) {
                // transpose order values that are less than the axis remains the same
                new_transpose_order[i] = order_val[j];
                ++j;
            } else if (order_val[j] == axis && static_cast<int>(k) <= n_axis_dims) {
                // these are new dims and they are not involved in the transposition. They have to stay in the same
                // place.
                new_transpose_order[i] = order_val[j] + k;
                ++k;
            } else {  // order_val[j] > axis
                /*
                    Transpose order values that are greater than the axis are shifted by N, where N is a count
                    of new added dimensions
                */
                new_transpose_order[i] = order_val[j] + n_axis_dims;
                ++j;
            }
        }
        auto new_order_const = ov::op::v0::Constant::create(transpose_order->get_element_type(),
                                                            {new_transpose_order.size()},
                                                            new_transpose_order);
        TransposeInputsInfo transpose_input_info = {transpose_info.transpose, new_order_const, 0};
        // deletes Transpose from 0 input
        auto success = sink_forward::UpdateInputTransposes(main_node, transpose_input_info, {0});
        if (!success) {
            return false;
        }
        auto new_axis = ov::op::v0::Constant::create(gather_axis->get_element_type(), gather_axis->get_shape(), {axis});
        main_node->input(2).replace_source_output(new_axis);
        copy_runtime_info(gather_axis, new_axis);

        default_outputs_update(main_node, transpose_input_info);
        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}

TSGatherBackward::TSGatherBackward() {
    MATCHER_SCOPE(TSGatherBackward);

    auto gather_label = wrap_type<ov::op::v8::Gather>({any_input(), any_input(), wrap_type<ov::op::v0::Constant>()},
                                                      CheckTransposeConsumers);
    auto transpose_label = wrap_type<ov::op::v1::Transpose>({gather_label, wrap_type<ov::op::v0::Constant>()},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
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
                        if (j >= new_shape.size() || shape[i] != new_shape[j]) {
                            if (shape[i] == 1) {
                                axes_val.push_back(i);
                                continue;
                            } else {
                                success = false;
                                break;
                            }
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
        std::vector<size_t> new_axes_val;
        if (!axes_val.empty()) {
            for (size_t i = 0; i < axes_val.size(); ++i) {
                new_axes_val.push_back(order_val[axes_val[i]]);
            }
            order_val = GetOrderAfterReduction(axes_val, order_val);
        }

        std::shared_ptr<ov::op::v0::Constant> new_axis;
        const auto& indices_rank_val = static_cast<size_t>(main_node->get_input_partial_shape(1).rank().get_length());

        std::vector<size_t> new_transpose_order;
        if (indices_rank_val > 0) {
            new_transpose_order.resize(order_val.size() - indices_rank_val + 1);

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
        } else {
            const std::vector<size_t> axes_values = {axis};
            new_transpose_order = GetOrderBeforeReduction(axes_values, order_val);
            new_axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, axis);
        }

        RemoveTransposeConsumers(main_node);
        if (success) {
            auto target_inputs = main_node->get_output_target_inputs(0);
            auto unsqueeze_axes = ov::op::v0::Constant::create(element::i32, {new_axes_val.size()}, new_axes_val);
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
        if (!new_axis) {
            new_axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, reversed_transpose_order[axis]);
        }
        copy_runtime_info(gather_axis, new_axis);
        main_node->input(2).replace_source_output(new_axis);
        main_node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
