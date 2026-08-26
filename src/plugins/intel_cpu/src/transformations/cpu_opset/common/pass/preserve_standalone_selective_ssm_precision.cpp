// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_standalone_selective_ssm_precision.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {
namespace {

bool is_selective_ssm(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::internal::SelectiveSSM>(node) || ov::is_type<ov::op::internal::PagedSelectiveSSM>(node);
}

bool is_paged_attention_model(const std::shared_ptr<ov::Model>& model) {
    const auto model_ops = model->get_ops();
    return std::any_of(model_ops.begin(), model_ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ov::op::PagedAttentionExtension>(node);
    });
}

}  // namespace

bool PreserveStandaloneSelectiveSSMPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (is_paged_attention_model(model)) {
        return false;
    }

    for (const auto& node : model->get_ordered_ops()) {
        if (!is_selective_ssm(node)) {
            continue;
        }

        ov::disable_conversion(node, ov::element::dynamic, ov::element::dynamic);
        for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
            ov::disable_conversion(node->get_input_node_shared_ptr(input_idx),
                                   ov::element::dynamic,
                                   ov::element::dynamic);
        }
    }

    return false;
}

bool RestoreStandalonePagedSelectiveSSMStatePrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (is_paged_attention_model(model)) {
        return false;
    }

    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(node);
        if (!paged_ssm) {
            continue;
        }

        const auto data_precision = paged_ssm->get_input_element_type(0);
        const auto state = ov::as_type_ptr<ov::op::v0::Parameter>(paged_ssm->get_input_node_shared_ptr(5));
        if (!state || data_precision.is_dynamic() || state->get_element_type() == data_precision) {
            continue;
        }

        state->set_element_type(data_precision);
        state->validate_and_infer_types();
        paged_ssm->validate_and_infer_types();
        changed = true;
    }

    return changed;
}

}  // namespace ov::intel_cpu
