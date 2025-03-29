// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_floatpoint_range.hpp"

#include "itt.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

void ov::pass::mark_range_path(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace("range_path", true);
}

bool ov::pass::is_range_path(const std::shared_ptr<const Node>& node) {
    return node->get_rt_info().count("range_path");
}

void ov::pass::erase_range_path(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase("range_path");
}

ov::pass::MarkFloatingPointRange::MarkFloatingPointRange() {
    MATCHER_SCOPE(MarkFloatingPointRange);
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;
    // through these nodes
    const auto range_propagating_nodes = pattern::wrap_type<ov::op::v0::Convert,
                                                            ov::op::v1::Greater,
                                                            ov::op::v1::GreaterEqual,
                                                            ov::op::v1::Less,
                                                            ov::op::v1::LessEqual,
                                                            ov::op::v1::Reshape,
                                                            ov::op::v4::Range,
                                                            ov::op::v0::Squeeze,
                                                            ov::op::v0::Unsqueeze>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!node)
            return false;

        auto range = ov::as_type_ptr<ov::op::v4::Range>(node);
        if (range && range->get_output_type().is_real()) {
            mark_range_path(node);
            ov::disable_fp16_compression(node);
            return true;
        }

        bool is_changed = false;

        for (const auto& in_node_output : node->input_values()) {
            auto input_node = in_node_output.get_node_shared_ptr();
            if (is_range_path(input_node)) {
                mark_range_path(node);
                ov::disable_fp16_compression(node);
                is_changed = true;
            }
        }

        return is_changed;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(range_propagating_nodes, matcher_name);
    register_matcher(m, callback);
}