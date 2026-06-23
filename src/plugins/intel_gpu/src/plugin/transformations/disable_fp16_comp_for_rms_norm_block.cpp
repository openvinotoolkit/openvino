// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_for_rms_norm_block.hpp"

#include "openvino/core/rt_info.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {

bool DisableFP16CompForRMSNormBlock::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool is_changed = false;

    for (const auto& node : model->get_ordered_ops()) {
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(node);
        if (!rms)
            continue;

        if (transformation_callback(rms))
            continue;

        if (ov::is_conversion_disabled(rms, ov::element::f16))
            continue;

        ov::disable_conversion(rms, ov::element::f16);
        is_changed = true;
    }

    return is_changed;
}

}  // namespace ov::intel_gpu
