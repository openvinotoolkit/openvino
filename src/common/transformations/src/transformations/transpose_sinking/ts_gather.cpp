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
    auto gather_label = wrap_type<Gather>({transpose_label, any_input(), any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = as_type_ptr<Transpose>(pattern_to_output.at(transpose_label));
        auto gather = pattern_to_output.at(gather_label);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        if (!transpose || !transpose_order) {
            return false;
        }
        TransposeInputsInfo transpose_input_info = {transpose, transpose_order, 0};
        sink_forward::UpdateInputTransposes(gather, transpose_input_info, {0, 1});

        auto axis = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int32_t>{0});
        const auto& indices = gather->input_value(3);
        auto new_axis = std::make_shared<Gather>(transpose_order, indices, axis);

        gather->input(3).replace_source_output(new_axis);
        gather->validate_and_infer_types();
        for (auto& new_node : sink_forward::InsertOutputTransposes(gather, transpose_input_info)) {
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

    auto gather_label = wrap_type<Gather>({any_input(), wrap_type<Constant>()}, HasSameOutputTransposeNodes);
    auto transpose_label = wrap_type<Transpose>({gather_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto gather = pattern_to_output.at(gather_label);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        if (!transpose_order) {
            return false;
        }

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(gather,
                                                                       transpose_order,
                /* input_indexes= */ {0, 1})) {
            register_new_node(new_node);
        }

        RemoveSingleOutputConsumers(gather);
        SwapNames(gather, transpose);

        const auto transpose_axis_order = transpose_order->get_axis_vector_val();
        const auto reversed_transpose_order = ReverseTransposeOrder(transpose_axis_order);
        auto axis = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int32_t>{0});
        auto data =
                std::make_shared<Constant>(element::i32, Shape{reversed_transpose_order.size()}, reversed_transpose_order);
        const auto& indices = gather->input_value(3);
        auto new_axis = std::make_shared<Gather>(data, indices, axis);
        gather->input(3).replace_source_output(new_axis);

        gather->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
