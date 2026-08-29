// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gqa.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/broadcast.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace attn {

namespace opp = ov::pass::pattern;

GQA::GQA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    // Match any GroupQueryAttention node regardless of input count (7 base + optional rotary inputs)
    auto gqa = opp::wrap_type<ov::op::internal::GroupQueryAttention>();

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_gqa = node_to_output.at(gqa).get_node_shared_ptr();

        LOG_DEBUG("GQA pattern matched: " << matched_gqa->get_friendly_name());

        auto isolate = [&](const std::shared_ptr<ov::Node>& node) {
            auto it = node_to_gptr->find(node);
            if (it != node_to_gptr->end()) {
                it->second->isolate(isol_tag);
            }
        };

        // Isolate a Slice node and its Broadcast+ShapeOf step-param harness (input[3]).
        auto isolate_slice_with_harness = [&](const std::shared_ptr<ov::Node>& slice) {
            isolate(slice);
            if (slice->get_input_size() > 3) {
                auto bcast = slice->input(3).get_source_output().get_node_shared_ptr();
                if (ov::is_type<ov::op::v3::Broadcast>(bcast)) {
                    isolate(bcast);
                    // ShapeOf feeds input[1] of Broadcast
                    if (bcast->get_input_size() > 1) {
                        auto shapeof = bcast->input(1).get_source_output().get_node_shared_ptr();
                        if (ov::is_type<ov::op::v3::ShapeOf>(shapeof)) {
                            isolate(shapeof);
                        }
                    }
                }
            }
        };

        // Isolate the GQA node itself
        isolate(matched_gqa);

        // input[4] (past_value) may pass through a Transpose before entering GQA.
        // Pull that Transpose into the same isolated block.
        if (matched_gqa->get_input_size() > 4) {
            auto src4 = matched_gqa->input(4).get_source_output().get_node_shared_ptr();
            if (ov::is_type<ov::op::v1::Transpose>(src4)) {
                isolate(src4);
            }
        }

        // output[1] (present_key): Slice -> Result (with Broadcast+ShapeOf harness).
        if (matched_gqa->get_output_size() > 1) {
            for (auto& target : matched_gqa->output(1).get_target_inputs()) {
                auto node = target.get_node()->shared_from_this();
                if (ov::is_type<ov::op::v8::Slice>(node)) {
                    isolate_slice_with_harness(node);
                }
            }
        }

        // output[2] (present_value): Slice -> Transpose -> Result (with Broadcast+ShapeOf harness).
        // Also handles bare Transpose -> Result when no Slice is present (no slice_kv).
        if (matched_gqa->get_output_size() > 2) {
            for (auto& target : matched_gqa->output(2).get_target_inputs()) {
                auto node = target.get_node()->shared_from_this();
                if (ov::is_type<ov::op::v8::Slice>(node)) {
                    isolate_slice_with_harness(node);
                    for (auto& t2 : node->output(0).get_target_inputs()) {
                        auto trans = t2.get_node()->shared_from_this();
                        if (ov::is_type<ov::op::v1::Transpose>(trans)) {
                            isolate(trans);
                        }
                    }
                } else if (ov::is_type<ov::op::v1::Transpose>(node)) {
                    // No Slice (slice_kv not used): bare Transpose on present_value.
                    isolate(node);
                }
            }
        }

        return false;
    };

    register_matcher(std::make_shared<opp::Matcher>(gqa, "TagGQA"), std::move(callback));
}

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
