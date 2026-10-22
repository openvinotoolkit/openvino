// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "keep_recurrent_cache_precision.hpp"
#include "openvino/op/recurrent_cache.hpp"
#include "openvino/op/recurrent_cache_update.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
namespace ov::intel_gpu {
bool KeepRecurrentCachePrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::internal::RecurrentGatedDeltaNetUpdate>(node) || ov::is_type<ov::op::internal::RecurrentGatedDeltaNet>(node)) {
            ov::mark_as_precision_sensitive(node->input(3));
            changed = true;
        }
        if (ov::is_type<ov::op::internal::RecurrentConvCacheUpdate>(node) || (ov::is_type<ov::op::internal::RecurrentConvCache>(node) && node->get_input_size() == 9)) {
            ov::mark_as_precision_sensitive(node->input(0));
            changed = true;
        }
    }
    return changed;
}
}  // namespace ov::intel_gpu
