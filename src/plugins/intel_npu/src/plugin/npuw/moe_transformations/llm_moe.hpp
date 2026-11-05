// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov::npuw::moe {

// Structural detection first; legacy router/expert node names cover older exports.
bool is_moe_model(const std::shared_ptr<ov::Model>& model);

enum class Stage { PREFILL, GENERATE };

// Adjust a stage's NPUW config for the given MoE hint.
void apply_moe_config(ov::AnyMap& stage_config, ::intel_npu::npuw::llm::MoEHint moe_hint, Stage stage);

// Collapse the tiled-MoE expert subgraph into MOECompressed ops (MoEHint::FUSED).
// Must run on a still-dynamic model, before ReshapeToStatic. Throws if nothing was fused.
void apply_moe_fusion(const std::shared_ptr<ov::Model>& model);

// True if every batched-MoE block in the model can be lowered to DEVICE_ROUTED, and there is at least one.
bool can_device_route_model(const std::shared_ptr<ov::Model>& model);

// Dynamic-shape phase: fuse only the stages whose hint is FUSED; other stages are left untouched.
void fuse_requested_stages(const ::intel_npu::Config& cfg,
                           const std::shared_ptr<ov::Model>& prefill_model,
                           const std::shared_ptr<ov::Model>& generate_model);

// Static-shape phase, for every stage: fold shape chains, resolve the hint and apply its config.
// FUSED and DEVICE_ROUTED are mutually exclusive per stage: a FUSED stage gets no routing model
// transformations here, and DEVICE_ROUTED can only be picked automatically while the generate hint is unset.
void configure_static_stages(::intel_npu::Config& cfg,
                             bool allow_auto_device_routed,
                             const std::shared_ptr<ov::Model>& prefill_model,
                             const std::vector<std::shared_ptr<ov::Model>>& generate_variants,
                             ov::AnyMap& prefill_config,
                             ov::AnyMap& generate_config);

}  // namespace ov::npuw::moe
