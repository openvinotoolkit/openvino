// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "untangle_dq_scale.hpp"

#include <numeric>

#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/util/op_types.hpp"

// Finds every Constant node that:
//   - is a tiny scalar (0-D or 1-element) of a floating-point type, and
//   - is consumed by more than one Multiply node,
// and gives each Multiply its own private copy.  This ensures that NPUW's FOLD
// pass sees one independent scalar bank entry per repeating-block instance.
//
// Implementation is a plain graph walk (like patterns::opt::untangleConst) to
// avoid any MatcherPass/GraphRewrite ordering subtleties.

bool ov::npuw::UntangleDQScale::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    // Snapshot the ops — we may add new Constants during the loop.
    const auto ops = model->get_ordered_ops();

    for (const auto& node : ops) {
        if (!ov::op::util::is_constant(node)) {
            continue;
        }

        const auto& shape = node->output(0).get_shape();
        const std::size_t total =
            std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
        if (total > 1) {
            continue;  // not a scalar — skip
        }

        const auto& etype = node->output(0).get_element_type();
        if (!etype.is_real()) {
            continue;  // only float scalars belong in the DQ scale bank
        }

        // Collect readers so we can iterate without invalidating the set.
        const auto readers_set = node->output(0).get_target_inputs();
        if (readers_set.size() <= 1) {
            continue;
        }

        // Clone the constant for every reader except the first, giving each
        // clone a name derived from the reader so get_unique_name() keeps them apart.
        auto it = readers_set.begin();
        ++it;  // skip first reader – it keeps the original constant
        auto const_node = std::static_pointer_cast<ov::op::v0::Constant>(node);
        for (; it != readers_set.end(); ++it) {
            auto cloned = std::make_shared<ov::op::v0::Constant>(*const_node);
            cloned->set_friendly_name(it->get_node()->get_friendly_name() + "/untangled_scale");
            it->replace_source_output(cloned->output(0));
            changed = true;
        }
    }

    return changed;
}
