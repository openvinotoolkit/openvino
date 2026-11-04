// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_weights_producer.hpp"

#include <cstdlib>
#include <string>
#include <utility>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "shared_weights_assigner.hpp"
#include "shared_weights_context_extractor.hpp"

namespace ov {
namespace intel_npu {
namespace transformations {

SharedWeightsResult assign_shared_weight_to_model_if_possible(const std::shared_ptr<ov::Model>& model,
                                                              const ov::Any& shared_weight_property) {
    OPENVINO_ASSERT(model, "Model for assigning shared weights must not be null");
    if (shared_weight_property.empty()) {
        return {};
    }

    OPENVINO_ASSERT(shared_weight_property.is<std::string>(), "NPU shared weight property must be a std::string");
    auto shared_device_contexts = ov::DeviceIDParser::get_hetero_devices(shared_weight_property.as<std::string>());
    SharedWeightsAssigner::Options options;
    options.shared_device_contexts = std::move(shared_device_contexts);
    options.preserve_weightless_cache_attr = (std::getenv("NO_WEIGHTLESS_ATTR") == nullptr);
    SharedWeightsAssigner assigner(std::move(options));
    auto collect_result = assigner.collect_and_partition(model);

    auto logger = ::intel_npu::Logger::global().clone("SharedWeightsProducer");
    logger.info("SHARED_WEIGHTS: %s", collect_result.statistic.to_string().c_str());

    auto shared_sources_with_constants =
        assigner.mutate_model_with_constant_sharing(std::move(collect_result.partitioned_constants));

    std::vector<std::shared_ptr<ov::AlignedBuffer>> shared_weight_sources;
    for (const auto& [shared_source, constants] : shared_sources_with_constants) {
        logger.info("SHARED_WEIGHTS: allocated shared source buffer: source_id: %zu, ptr: %p, size: %zu, "
                    "holds shared weight count: %zu",
                    shared_source->get_descriptor()->get_id(),
                    static_cast<void*>(shared_source->get_ptr<char>()),
                    shared_source->size(),
                    constants.size());
        shared_weight_sources.push_back(shared_source);
    }

    auto shared_ctx_ptr = SharedWeightsContextExtractor::extract_weight_sharing_context(shared_sources_with_constants);
    return {std::move(shared_weight_sources), std::move(shared_ctx_ptr)};
}

}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
