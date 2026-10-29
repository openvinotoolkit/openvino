// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "keep_recurrent_cache_precision.hpp"
#include "openvino/op/recurrent_cache_update.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/op/reshape.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
namespace ov::intel_gpu {
bool KeepRecurrentCachePrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    auto protect_cache = [](const ov::Input<ov::Node>& input) {
        auto source = input.get_source_output().get_node_shared_ptr();
        auto root = source;
        while (ov::is_type<ov::op::v1::Reshape>(root))
            root = root->input_value(0).get_node_shared_ptr();
        // Protect aliases of caller-owned cache Parameters. Temporary snapshot
        // banks are computation results: their precision must follow the graph,
        // including ordinary scatter writeback into a compressed destination.
        if (!ov::is_type<ov::op::v0::Parameter>(root))
            return;
        ov::mark_as_precision_sensitive(input);
        // llama.cpp flattens its FP32 cache through Reshapes. Protect those
        // aliases too; marking only the consumer permits full-cache FP16 reorders.
        while (ov::is_type<ov::op::v1::Reshape>(source)) {
            ov::disable_conversion(source, ov::element::f16);
            ov::mark_as_precision_sensitive(source->input(0));
            source = source->input_value(0).get_node_shared_ptr();
        }
    };
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::internal::RecurrentGatedDeltaNetUpdate>(node)) {
            protect_cache(node->input(3));
            changed = true;
        }
        if (ov::is_type<ov::op::internal::RecurrentConvCacheUpdate>(node)) {
            protect_cache(node->input(0));
            changed = true;
        }
    }
    return changed;
}
}  // namespace ov::intel_gpu
