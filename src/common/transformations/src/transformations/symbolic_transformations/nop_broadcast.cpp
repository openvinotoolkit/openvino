// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/nop_broadcast.hpp"

#include <openvino/core/dimension_tracker.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace {
auto bc_predicate = [](Output<Node> output) {
    const auto& op = output.get_node_shared_ptr();
    auto data_rank = op->get_input_partial_shape(0).rank();
    auto new_shape_shape = op->get_input_partial_shape(1);
    return data_rank.is_static() && new_shape_shape.is_static() && data_rank == new_shape_shape[0];
};
}

ov::pass::NopBroadcast::NopBroadcast() {
    MATCHER_SCOPE(NopBroadcast);
    auto input_label = pattern::any_input(pattern::has_static_rank());
    auto shape_of = pattern::wrap_type<op::v0::ShapeOf, op::v3::ShapeOf>();
    auto ones = INT_CONSTANT_WITH_PREDICATE(std::all_of(value.begin(), value.end(), cmp::Equal<int64_t>(1)));
    auto maximum = pattern::wrap_type<op::v1::Maximum>({shape_of, ones});
    auto broadcast_three_inputs =
        pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({input_label, maximum, pattern::any_input()},
                                                                 bc_predicate);
    auto broadcast_two_inputs =
        pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({input_label, maximum}, bc_predicate);
    auto or_label = make_shared<pattern::op::Or>(OutputVector{broadcast_two_inputs, broadcast_three_inputs});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& vm = m.get_pattern_value_map();
        auto data = vm.at(input_label);
        auto shape = vm.at(shape_of);

        ov::TensorLabel data_labels, shape_labels;
        if (!get_labels(data.get_partial_shape(), data_labels) || !get_labels(shape, shape_labels))
            return false;

        if (!are_unique_and_equal_labels(data_labels, shape_labels))
            return false;
        return ov::replace_output_update_name(m.get_match_root(), data);
    };

    auto m = std::make_shared<pattern::Matcher>(or_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
