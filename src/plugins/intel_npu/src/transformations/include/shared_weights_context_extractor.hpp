// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
namespace intel_npu {
namespace transformations {

/**
 * @brief Builds a weight-sharing context from a model or from newly created shared weight sources.
 */
struct SharedWeightsContextExtractor {
    /// Pointer to an OpenVINO weight-sharing context.
    using WeightSharingContextPtr = std::shared_ptr<ov::weight_sharing::Context>;
    /// Model constant backed by shared storage.
    using SharedConstant = std::shared_ptr<ov::op::v0::Constant>;
    /// A shared source and the constants backed by it.
    using SharedSourceAndConstants = std::pair<std::shared_ptr<ov::AlignedBuffer>, std::vector<SharedConstant>>;
    /// Shared sources paired with their constants.
    using SharedSourcesWithConstants = std::vector<SharedSourceAndConstants>;

    /// Extracts the weight registry and runtime sources associated with a model.
    /// @param model Model containing the weight-sharing metadata.
    /// @return A context populated from the model's weight registry and runtime sources.
    static WeightSharingContextPtr extract_weight_sharing_context(const std::shared_ptr<ov::Model>& model);

    /// Creates a context that registers the supplied shared sources and their constants.
    /// @param collected_sources_with_constants Shared source buffers paired with their constants.
    /// @return A context populated with the supplied sources and constants.
    static WeightSharingContextPtr extract_weight_sharing_context(
        const SharedSourcesWithConstants& collected_sources_with_constants);
};

}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
