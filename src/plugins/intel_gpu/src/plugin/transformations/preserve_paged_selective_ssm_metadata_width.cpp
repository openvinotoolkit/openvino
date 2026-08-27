// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_paged_selective_ssm_metadata_width.hpp"

#include <utility>

#include "intel_gpu/primitives/paged_selective_ssm.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/paged_selective_ssm.hpp"

namespace ov::intel_gpu {

class OriginalPagedSelectiveSSMMetadataInputs final : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("intel_gpu::OriginalPagedSelectiveSSMMetadataInputs", "0", ov::RuntimeAttribute);

    explicit OriginalPagedSelectiveSSMMetadataInputs(ov::OutputVector inputs) : m_inputs(std::move(inputs)) {}

    [[nodiscard]] bool is_copyable() const override {
        return false;
    }

    const ov::OutputVector& get_inputs() const {
        return m_inputs;
    }

private:
    ov::OutputVector m_inputs;
};

namespace {

constexpr size_t metadata_first_input = cldnn::paged_selective_ssm::SUBSEQUENCE_BEGINS;
constexpr size_t metadata_input_count = cldnn::paged_selective_ssm::CACHE_INTERVAL - metadata_first_input + 1;

}  // namespace

bool RecordPagedSelectiveSSMMetadataInputs::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(node);
        if (!paged_ssm || paged_ssm->get_input_element_type(metadata_first_input) != ov::element::i64) {
            continue;
        }

        auto& rt_info = paged_ssm->get_rt_info();
        if (rt_info.count(OriginalPagedSelectiveSSMMetadataInputs::get_type_info_static()) != 0) {
            continue;
        }

        ov::OutputVector metadata_inputs;
        metadata_inputs.reserve(metadata_input_count);
        for (size_t input_index = metadata_first_input; input_index <= cldnn::paged_selective_ssm::CACHE_INTERVAL; ++input_index) {
            metadata_inputs.push_back(paged_ssm->input_value(input_index));
        }
        rt_info[OriginalPagedSelectiveSSMMetadataInputs::get_type_info_static()] = OriginalPagedSelectiveSSMMetadataInputs{std::move(metadata_inputs)};
        changed = true;
    }
    return changed;
}

bool PreservePagedSelectiveSSMMetadataWidth::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(node);
        if (!paged_ssm) {
            continue;
        }

        auto& rt_info = paged_ssm->get_rt_info();
        const auto original_inputs_it = rt_info.find(OriginalPagedSelectiveSSMMetadataInputs::get_type_info_static());
        if (original_inputs_it == rt_info.end()) {
            continue;
        }

        const auto original_inputs = original_inputs_it->second.as<OriginalPagedSelectiveSSMMetadataInputs>().get_inputs();
        OPENVINO_ASSERT(original_inputs.size() == metadata_input_count);
        rt_info.erase(original_inputs_it);

        bool node_changed = false;
        for (size_t metadata_index = 0; metadata_index < original_inputs.size(); ++metadata_index) {
            const size_t input_index = metadata_first_input + metadata_index;
            if (paged_ssm->input_value(input_index) == original_inputs[metadata_index]) {
                continue;
            }

            paged_ssm->input(input_index).replace_source_output(original_inputs[metadata_index]);
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
