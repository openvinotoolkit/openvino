// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_selective_ssm_precision.hpp"

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {

bool PreserveSelectiveSSMPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::internal::SelectiveSSM>(node) &&
            !ov::is_type<ov::op::internal::PagedSelectiveSSM>(node)) {
            continue;
        }

        ov::disable_conversion(node, ov::element::dynamic, ov::element::dynamic);
        for (size_t input = 0; input < 6; ++input) {
            ov::disable_conversion(node->get_input_node_shared_ptr(input),
                                   ov::element::dynamic,
                                   ov::element::dynamic);
        }
    }
    return false;
}

}  // namespace ov::intel_gpu
