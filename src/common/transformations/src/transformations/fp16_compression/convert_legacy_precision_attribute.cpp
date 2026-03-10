// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/convert_legacy_precision_attribute.hpp"

#include "itt.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

bool ov::pass::ConvertLegacyPrecisionAttribute::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(ConvertLegacyPrecisionAttribute);
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        if (fp16_compression_is_disabled(node)) {
            enable_fp16_compression(node);  // remove legacy attribute
            disable_compression_to(node, element::f16);
            changed = true;
        }
        OPENVINO_SUPPRESS_DEPRECATED_END

        if (auto sub_graph_node = ov::as_type_ptr<op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < sub_graph_node->get_internal_subgraphs_size(); ++i) {
                changed = run_on_model(sub_graph_node->get_function(static_cast<int>(i))) || changed;
            }
        }
    }
    return changed;
}
