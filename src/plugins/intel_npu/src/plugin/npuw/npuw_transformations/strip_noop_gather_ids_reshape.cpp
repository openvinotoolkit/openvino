// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strip_noop_gather_ids_reshape.hpp"

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"

namespace {

// Matches the threshold of the host gather: smaller embeddings are not worth moving to the host.
constexpr size_t MIN_HIDDEN_SIZE = 2048;

ov::Output<ov::Node> strip_noop_reshape_squeeze(const ov::Output<ov::Node>& ids) {
    const auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(ids.get_node_shared_ptr());
    if (!squeeze) {
        return ids;
    }
    const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(squeeze->input_value(0).get_node_shared_ptr());
    if (!reshape) {
        return ids;
    }
    const auto source = reshape->input_value(0);
    const bool same_shape = source.get_partial_shape().is_static() && ids.get_partial_shape().is_static() &&
                            source.get_shape() == ids.get_shape();
    return same_shape ? source : ids;
}

bool is_vocab_gather(const std::shared_ptr<ov::op::v8::Gather>& gather) {
    if (!gather || gather->get_input_partial_shape(0).is_dynamic()) {
        return false;
    }
    const auto& vocab_shape = gather->get_input_shape(0);
    return vocab_shape.size() == 2 && vocab_shape[1] >= MIN_HIDDEN_SIZE;
}

}  // namespace

bool ov::npuw::StripNoopGatherIdsReshape::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node);
        if (!is_vocab_gather(gather)) {
            continue;
        }
        const auto ids = gather->input_value(1);
        const auto stripped = strip_noop_reshape_squeeze(ids);
        if (stripped != ids) {
            gather->input(1).replace_source_output(stripped);
            changed = true;
        }
    }
    return changed;
}
