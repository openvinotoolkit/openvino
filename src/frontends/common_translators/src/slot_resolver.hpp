// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace ov {
namespace frontend {
namespace pass {
namespace sal_detail {

using Slots = std::vector<ov::Output<ov::Node>>;

struct LengthTemplate {
    int64_t length{};
    std::vector<ov::Output<ov::Node>> slot_templates;
};

// Shape-preserving If-based select. Unlike v1::Select, does not broadcast branch
// inputs - used when branches may have legitimately different shapes.
ov::Output<ov::Node> make_shape_preserving_select(const ov::Output<ov::Node>& cond_in,
                                                  const ov::Output<ov::Node>& then_val,
                                                  const ov::Output<ov::Node>& else_val);

// Strip a chain of v16::Identity wrappers, returning the underlying value.
ov::Output<ov::Node> unwrap_identity(const ov::Output<ov::Node>& value);

// Build a zero-filled Constant whose static dims match the template (dynamic dims -> 0).
ov::Output<ov::Node> make_zero_dummy(const ov::Output<ov::Node>& tmpl);

// Seed constant with static dims preserved, last dynamic axis set to 0
// (growable concat axis), all other dynamic axes set to 1.
ov::Output<ov::Node> make_growable_seed(const ov::PartialShape& ps, ov::element::Type et);

// Maps each sequence-typed value to its vector of per-element "slot" tensors.
// Crossing a MultiSubGraphOp (Loop/If) boundary, a sequence carried across
// iterations is expanded into N parallel tensor back-edges:
//
//   Loop( seqParam --..append..-- seqResult )      1 sequence carrier
//             ||
//             vv
//   Loop( s0Param --..-- s0Result )                N tensor carriers
//       ( s1Param --..-- s1Result )
//       ( ...                     )
class SlotResolver {
public:
    explicit SlotResolver(const std::shared_ptr<ov::Model>& root) : root_(root) {
        build_maps(root_);
        preallocate_loop_merged_params(root_);
    }

    bool changed() const {
        return changed_;
    }

    std::optional<Slots> slots_of(const ov::Output<ov::Node>& value_in);
    std::optional<int64_t> length_of(const ov::Output<ov::Node>& value_in);

    // Resolve outer-seed and back-edge slots for every pre-allocated Loop
    // merged-input entry and call set_merged_input. Must be invoked after
    // all helper replacements have been performed (so the chains are stable)
    // and before the final graph cleanup.
    void finalize_pending_wiring();

private:
    struct PendingMerged {
        std::shared_ptr<ov::op::v5::Loop> loop;
        std::shared_ptr<ov::Model> body;
        int back_edge_result_idx{-1};
        int outer_input{-1};
        std::shared_ptr<ov::op::v0::Parameter> old_param;
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_params;
        Slots outer_seed_slots;  // pre-resolved seed slots when known
        bool outer_seed_resolved{false};
    };

    void build_maps(const std::shared_ptr<ov::Model>& m);
    void preallocate_loop_merged_params(const std::shared_ptr<ov::Model>& m);
    bool is_sequence_merged_input(const std::shared_ptr<ov::op::v5::Loop>& loop,
                                  const std::shared_ptr<ov::Model>& body,
                                  size_t p_idx,
                                  size_t back_value_idx);
    std::optional<Slots> slots_of_param(const std::shared_ptr<ov::op::v0::Parameter>& p);
    std::optional<Slots> slots_of_msg_output(const std::shared_ptr<ov::op::util::MultiSubGraphOp>& msg, size_t out_idx);
    std::optional<LengthTemplate> find_template_via_chain(const ov::Output<ov::Node>& root_value, ov::Node* exclude_p);

    std::shared_ptr<ov::Model> root_;
    std::vector<PendingMerged> pending_merged_;
    std::map<ov::Output<ov::Node>, Slots> cache_;
    std::set<ov::Output<ov::Node>> in_progress_;
    std::map<ov::op::v0::Parameter*, std::shared_ptr<ov::Model>> param_to_model_;
    std::map<ov::Model*, std::pair<std::shared_ptr<ov::op::util::MultiSubGraphOp>, int>> body_owner_;
    bool changed_ = false;
};

}  // namespace sal_detail
}  // namespace pass
}  // namespace frontend
}  // namespace ov
