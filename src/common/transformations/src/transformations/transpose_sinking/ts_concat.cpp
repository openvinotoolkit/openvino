// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_concat.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

TSConcatForward::TSConcatForward() {
    MATCHER_SCOPE(TSConcatForward);

    create_pattern<ov::op::v0::Concat>();

    auto sinking_transformation = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<Node>& main_node,
                                                            const TransposeInputsInfo& transpose_info) -> bool {
        // todo: support dynamic rank case
        auto concat_node = as_type_ptr<ov::op::v0::Concat>(main_node);
        if (!concat_node) {
            return false;
        }

        if (transformation_callback(concat_node)) {
            return false;
        }

        auto concat_axis = concat_node->get_axis();
        if (concat_axis < 0) {
            if (concat_node->get_output_partial_shape(0).rank().is_dynamic()) {
                return false;
            }
            const auto rank = concat_node->get_output_partial_shape(0).rank().get_length();
            concat_axis = ov::util::normalize(concat_axis, rank);
        }

        // todo: support dyn rank case
        bool updated = sink_forward::UpdateInputTransposes(main_node, transpose_info);
        if (!updated) {
            return false;
        }

        const auto transpose_axis_order = transpose_info.transpose_const->get_axis_vector_val();
        const int64_t transposed_concat_axis = transpose_axis_order[concat_axis];
        concat_node->set_axis(transposed_concat_axis);

        default_outputs_update(main_node, transpose_info);
        return true;
    };
    transpose_sinking(matcher_name, sinking_transformation);
}

TSConcatBackward::TSConcatBackward() {
    MATCHER_SCOPE(TSConcatBackward);

    auto main_node_label = wrap_type<ov::op::v0::Concat>([](const Output<Node>& output) -> bool {
        return has_static_rank()(output) && CheckTransposeConsumers(output);
    });

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({main_node_label, transpose_const_label},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();
        if (transformation_callback(main_node)) {
            return false;
        }

        auto concat_node = as_type_ptr<ov::op::v0::Concat>(main_node);
        if (!concat_node) {
            return false;
        }

        auto concat_axis = concat_node->get_axis();
        if (concat_axis < 0) {
            if (concat_node->get_output_partial_shape(0).rank().is_dynamic()) {
                return false;
            }

            const auto rank = concat_node->get_output_partial_shape(0).rank().get_length();
            concat_axis = ov::util::normalize(concat_axis, rank);
        }

        const auto transpose_axis_order = transpose_const->get_axis_vector_val();
        const auto reversed_transpose_axis_order = ReverseTransposeOrder(transpose_axis_order);
        if (static_cast<int64_t>(reversed_transpose_axis_order.size()) <= concat_axis) {
            return false;
        }

        const auto transposed_concat_axis = reversed_transpose_axis_order[concat_axis];
        concat_node->set_axis(static_cast<int64_t>(transposed_concat_axis));

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, transpose_const)) {
            register_new_node(new_node);
        }
        concat_node->validate_and_infer_types();

        RemoveTransposeConsumers(main_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
