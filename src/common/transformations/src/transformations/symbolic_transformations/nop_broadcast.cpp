// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/nop_broadcast.hpp"

#include <openvino/core/dimension_tracker.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "itt.hpp"

ov::pass::NopBroadcast::NopBroadcast() {
    MATCHER_SCOPE(NopBroadcast);
    auto input_label = pattern::any_input(pattern::has_static_rank());
    auto shape_of = pattern::wrap_type<op::v0::ShapeOf, op::v3::ShapeOf>();
    auto ones = pattern::wrap_type<op::v0::Constant>();
    auto maximum = pattern::wrap_type<op::v1::Maximum>({shape_of, ones});
    auto broadcast = pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({input_label, maximum});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto constant = ov::as_type_ptr<op::v0::Constant>(pattern_to_output.at(ones).get_node_shared_ptr());
        if (!constant)
            return false;
        auto valid_labels = [](const ov::TensorLabel& labels) {
            return !labels.empty() && std::all_of(labels.begin(), labels.end(), [](const label_t& l) {
                return l != 0;
            });
        };
        auto shape_of_labels = pattern_to_output.at(shape_of).get_tensor().get_value_label();
        if (!valid_labels(shape_of_labels))
            return false;
        auto input = pattern_to_output.at(input_label);
        ov::TensorLabel input_labels;
        for (const auto& dim : input.get_partial_shape()) {
            if (dim.get_max_length() == 0)
                return false;
            input_labels.push_back(ov::DimensionTracker::get_label(dim));
        }
        if (!valid_labels(input_labels))
            return false;
        auto constant_content = constant->cast_vector<int64_t>();
        bool all_ones = std::all_of(constant_content.begin(), constant_content.end(), [](const int64_t& i) {
            return i == 1;
        });
        if (constant_content.size() > input.get_partial_shape().size() || !all_ones)
            return false;
        auto output = pattern_to_output.at(broadcast);
        output.replace(input);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
