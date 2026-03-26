// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "apply_moe_device_routed_transforms.hpp"

#include "../logging.hpp"
#include "../moe_transformations/device_routed_moe_transform.hpp"
#include "../moe_transformations/gather_to_2d_gather.hpp"

namespace {

// Apply DEVICE_ROUTED MoE transformations to models
void apply_moe_device_routed_transforms(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::pass::DeviceRoutedMoETransform moe_transform;
    ov::npuw::pass::GatherTo2DGather gather_transform;

    moe_transform.run_on_model(model);
    LOG_DEBUG("  Applied DEVICE_ROUTED transformations to model variant");

    // Apply Gather to 2D Gather transformation for HW optimization
    gather_transform.run_on_model(model);
    LOG_DEBUG("  Applied GatherTo2DGather transformation to model variant");
}

}  // namespace

bool ApplyMoEDeviceRoutedTransforms::run_on_model(const std::shared_ptr<ov::Model>& model) {
    apply_moe_device_routed_transforms(model);

    return true;
}
