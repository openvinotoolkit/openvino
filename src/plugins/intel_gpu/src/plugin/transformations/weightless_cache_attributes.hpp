// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <utility>

#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

using WeightlessConstantMap =
    std::unordered_map<size_t, std::pair<std::shared_ptr<ov::op::v0::Constant>, ov::WeightlessCacheAttribute>>;

/**
 * Records original constants before OTD graph transformations. Constants with shared
 * source descriptors receive WCA, and constants with WCA receive a copyable
 * otd_bin_offset marker. The shared map retains their storage and original metadata
 * for RestoreWeightlessCacheAttributes; it must be local to one pipeline invocation.
 * Constant[WCA] -> Constant[WCA, otd_bin_offset], with unchanged tensor data.
 */
class MarkWeightlessCacheAttributes : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MarkWeightlessCacheAttributes");
    explicit MarkWeightlessCacheAttributes(std::shared_ptr<WeightlessConstantMap> original_constants);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<WeightlessConstantMap> m_original_constants;
};

/**
 * Restores WCA after OTD graph transformations, using the map populated by
 * MarkWeightlessCacheAttributes. Only constants without WCA and with a recorded
 * offset, matching element type, byte size and exact bytes are eligible. Shared
 * data pointers skip the byte comparison. Shape-only changes are allowed.
 * Constant[otd_bin_offset] -> Constant[otd_bin_offset, WCA], without changing data.
 */
class RestoreWeightlessCacheAttributes : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RestoreWeightlessCacheAttributes");
    explicit RestoreWeightlessCacheAttributes(std::shared_ptr<const WeightlessConstantMap> original_constants);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<const WeightlessConstantMap> m_original_constants;
};

}  // namespace ov::intel_gpu
