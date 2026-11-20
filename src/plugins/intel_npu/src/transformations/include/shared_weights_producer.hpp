// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <vector>

namespace ov {
class AlignedBuffer;
class Any;
class Model;
namespace weight_sharing {
struct Context;
}
namespace intel_npu {
namespace transformations {

/// Shared source buffers and the weight-sharing context that registers them.
using SharedWeightsResult =
    std::tuple<std::vector<std::shared_ptr<ov::AlignedBuffer>>, std::shared_ptr<ov::weight_sharing::Context>>;

/**
 * @brief Assigns shareable model constants to shared weight sources when the property is configured.
 *
 * An empty property leaves the model unchanged and returns an empty result. Otherwise, the property must contain
 * a string describing the shared device contexts.
 * @param model Model whose constants may be moved to shared sources.
 * @param shared_weight_property Property describing the shared device contexts; an empty property disables sharing.
 * @return The allocated shared source buffers and a context registering their constants.
 */
SharedWeightsResult assign_shared_weight_to_model_if_possible(const std::shared_ptr<ov::Model>& model,
                                                              const ov::Any& shared_weight_property);

}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
