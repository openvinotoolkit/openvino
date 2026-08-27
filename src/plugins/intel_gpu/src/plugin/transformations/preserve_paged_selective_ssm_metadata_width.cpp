// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_paged_selective_ssm_metadata_width.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/paged_selective_ssm.hpp"

namespace ov::intel_gpu {

bool PreservePagedSelectiveSSMMetadataWidth::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(node);
        if (!paged_ssm) {
            continue;
        }

        bool node_changed = false;
        for (size_t input_index = 6; input_index < paged_ssm->get_input_size(); ++input_index) {
            const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(paged_ssm->get_input_node_shared_ptr(input_index));
            if (!convert || convert->get_input_element_type(0) != ov::element::i64 || convert->get_output_element_type(0) != ov::element::i32) {
                continue;
            }

            paged_ssm->input(input_index).replace_source_output(convert->input_value(0));
            node_changed = true;
        }

        if (node_changed) {
            paged_ssm->validate_and_infer_types();
            changed = true;
        }
    }
    return changed;
}

}  // namespace ov::intel_gpu
