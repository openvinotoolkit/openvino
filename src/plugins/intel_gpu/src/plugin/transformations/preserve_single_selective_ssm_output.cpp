// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_single_selective_ssm_output.hpp"

#include <string>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov::intel_gpu {

bool EliminateEmptySelectiveSSM::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto selective_ssm = ov::as_type_ptr<ov::op::internal::SelectiveSSM>(node);
        if (!selective_ssm) {
            continue;
        }

        const auto& x_shape = selective_ssm->get_input_partial_shape(3);
        if (x_shape.rank().is_dynamic() || x_shape.size() < 2 || x_shape[1].is_dynamic() || x_shape[1].get_length() != 0) {
            continue;
        }

        selective_ssm->output(0).replace(selective_ssm->input_value(3));
        selective_ssm->output(1).replace(selective_ssm->input_value(5));
        changed = true;
    }
    return changed;
}

bool PreserveSingleSelectiveSSMOutput::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto selective_ssm = ov::as_type_ptr<ov::op::internal::SelectiveSSM>(node);
        if (!selective_ssm) {
            continue;
        }

        const bool output0_used = !selective_ssm->output(0).get_target_inputs().empty();
        const bool output1_used = !selective_ssm->output(1).get_target_inputs().empty();
        if (output0_used == output1_used) {
            continue;
        }

        const size_t output_index = output1_used ? 1 : 0;
        if (!selective_ssm->is_dynamic() && output_index == 0) {
            continue;
        }

        auto output = selective_ssm->output(output_index);
        const auto target_inputs = output.get_target_inputs();
        const auto output_shape = std::make_shared<ov::op::v3::ShapeOf>(output, ov::element::i64);
        const auto output_view = std::make_shared<ov::op::v1::Reshape>(output, output_shape, false);
        output_view->set_friendly_name(selective_ssm->get_friendly_name() + "/output_view_" + std::to_string(output_index));
        output_view->output(0).get_tensor().set_names(output.get_names());
        ov::copy_runtime_info(selective_ssm, {output_shape, output_view});
        for (const auto& input : target_inputs) {
            input.replace_source_output(output_view);
        }
        changed = true;
    }
    return changed;
}

}  // namespace ov::intel_gpu
