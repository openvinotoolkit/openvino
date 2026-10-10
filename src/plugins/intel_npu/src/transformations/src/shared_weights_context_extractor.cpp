// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_weights_context_extractor.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/weight_sharing_util.hpp"

namespace ov {
namespace intel_npu {
namespace transformations {

SharedWeightsContextExtractor::WeightSharingContextPtr SharedWeightsContextExtractor::extract_weight_sharing_context(
    const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model && "Model for extracting weight sharing context must not be null");

    auto context = std::make_shared<ov::weight_sharing::Context>();
    context->m_weight_registry = ov::weight_sharing::Extension::get_weight_registry(*model);
    context->m_runtime_sources = ov::weight_sharing::Extension::get_weight_sources(*model);
    return context;
}

SharedWeightsContextExtractor::WeightSharingContextPtr SharedWeightsContextExtractor::extract_weight_sharing_context(
    const SharedSourcesWithConstants& collected_sources_with_constants) {
    auto context = std::make_shared<ov::weight_sharing::Context>();
    for (const auto& [source, constants] : collected_sources_with_constants) {
        ov::weight_sharing::set_runtime_weight_source(*context, source);
        for (const auto& constant : constants) {
            ov::weight_sharing::set_constant(*context, *constant);
        }
    }
    return context;
}
}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
