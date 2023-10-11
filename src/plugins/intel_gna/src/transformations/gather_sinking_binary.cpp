// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_binary.hpp"

#include <transformations/utils/utils.hpp>
#include <utility>

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace gather_sinking;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::rt_info;
using namespace ov::intel_gna::pass::helper;
using namespace ov::intel_gna::graph_utils;

GatherSinkingBinaryForward::GatherSinkingBinaryForward() {
    MATCHER_SCOPE(GatherSinkingBinaryForward);

    auto if_gather_has_constants_rank_not_more_than_one = [](const GatherInputsInfo& inputs_info) -> bool {
        return constant_has_rank_not_more_than(inputs_info.axis_const, 1) &&
               constant_has_rank_not_more_than(inputs_info.indices_const, 1);
    };

    auto main_node_label = wrap_type<ov::op::util::BinaryElementwiseArithmetic>(
        [if_gather_has_constants_rank_not_more_than_one](const Output<Node>& output) -> bool {
            return if_node_has_gather_inputs(output, if_gather_has_constants_rank_not_more_than_one);
        });

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& main_node_output = pattern_to_output.at(main_node_label);
        auto main_node = main_node_output.get_node_shared_ptr();

        GatherInputsInfo gather_input_info = get_first_gather_input(main_node);

        sink_forward::update_input_gather(main_node, gather_input_info);
        for (auto& new_node : sink_forward::insert_output_gather(main_node, gather_input_info)) {
            register_new_node(new_node);
            gather_sinking::update_forward_gather_sinking_ability(new_node);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

GatherSinkingBinaryBackward::GatherSinkingBinaryBackward() {
    MATCHER_SCOPE(GatherSinkingBinaryBackward);
    auto main_node_label = wrap_type<ov::op::util::BinaryElementwiseArithmetic>([](const Output<Node>& output) -> bool {
        return has_static_rank()(output) && has_same_output_gather_nodes(output);
    });

    auto indices_const_label = wrap_type<Constant>(rank_not_more_than(1));
    auto axes_const_label = wrap_type<Constant>(rank_not_more_than(1));

    auto gather_label = wrap_type<Gather>({main_node_label, indices_const_label, axes_const_label},
                                          [](const Output<Node>& output) -> bool {
                                              return has_static_rank()(output) && is_gather_sinking_node(output);
                                          });

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto indices_const = as_type_ptr<Constant>(pattern_to_output.at(indices_const_label).get_node_shared_ptr());
        auto axes_const = as_type_ptr<Constant>(pattern_to_output.at(axes_const_label).get_node_shared_ptr());
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();

        for (auto& new_node : sink_backward::insert_gather_before_node(main_node, indices_const, axes_const, gather)) {
            register_new_node(new_node);
        }

        // remove output transposes
        remove_single_output_consumers(main_node);

        swap_names(gather, main_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
