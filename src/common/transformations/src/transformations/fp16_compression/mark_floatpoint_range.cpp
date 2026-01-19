// Copyright (C) 2018-2026 Intel Corporation
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
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;

namespace ov::pass {

void mark_range_path(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace("range_path", true);
}

bool is_range_path(const std::shared_ptr<const Node>& node) {
    return node->get_rt_info().count("range_path");
}

void erase_range_path(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase("range_path");
}

MarkFloatingPointRange::MarkFloatingPointRange() {
    MATCHER_SCOPE(MarkFloatingPointRange);
    // through these nodes
    const auto range_propagating_nodes = pattern::wrap_type<v0::Convert,
                                                            v1::Greater,
                                                            v1::GreaterEqual,
                                                            v1::Less,
                                                            v1::LessEqual,
                                                            v1::Reshape,
                                                            v4::Range,
                                                            v0::Squeeze,
                                                            v0::Unsqueeze>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!node)
            return false;

        bool is_changed = false;

        auto range = ov::as_type_ptr<v4::Range>(node);
        if (range && range->get_output_type().is_real()) {
            mark_range_path(node);
            ov::disable_fp16_compression(node);

            // mark inputs as well
            for (const auto& range_input : range->input_values()) {
                ov::disable_fp16_compression(range_input.get_node_shared_ptr());
            }
            is_changed = true;
        } else {
            for (const auto& in_node_output : node->input_values()) {
                auto input_node = in_node_output.get_node_shared_ptr();

                if (is_range_path(input_node)) {
                    mark_range_path(node);
                    ov::disable_fp16_compression(node);
                    is_changed = true;
                    break;
                }
            }
        }

        return is_changed;
    };
    auto m = std::make_shared<pattern::Matcher>(range_propagating_nodes, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
