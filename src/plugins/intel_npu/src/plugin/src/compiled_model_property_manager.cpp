// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model_property_manager.hpp"

#include <sstream>

#include "metadata.hpp"

namespace intel_npu {

CompiledModelPropertyManager::CompiledModelPropertyManager(const FilteredConfig& config,
                                                           const std::shared_ptr<IGraph>& graph,
                                                           const std::optional<int64_t>& batchSize,
                                                           Logger& logger)
    : _graph(graph),
      _batchSize(batchSize),
      _logger(logger),
      _properties(PropertiesType::COMPILED_MODEL, config) {}

void CompiledModelPropertyManager::setProperty(const ov::AnyMap& properties) {
    _properties.setProperty(properties);
}

ov::Any CompiledModelPropertyManager::getProperty(const std::string& name) const {
    if (name == ov::model_name.name()) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return _graph->get_metadata().name;
    }

    if (name == ov::runtime_requirements.name()) {
        _properties.getProperty(name);
        return buildRuntimeRequirements();
    }

    return _properties.getProperty(name);
}

std::string CompiledModelPropertyManager::buildRuntimeRequirements() const {
    OPENVINO_ASSERT(_graph != nullptr, "Missing graph");

    auto compatibilityDescriptor = _graph->get_compatibility_descriptor();
    if (compatibilityDescriptor.has_value()) {
        const auto descriptorView = compatibilityDescriptor.value();
        _logger.debug("Runtime requirements from the graph %.*s length: %zu",
                      static_cast<int>(descriptorView.size()),
                      descriptorView.data(),
                      descriptorView.size());
    }

    std::ostringstream requirementsString;
    Metadata<CURRENT_METADATA_VERSION>(
        0,
        CURRENT_OPENVINO_VERSION,
        std::nullopt,
        _batchSize,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        compatibilityDescriptor)
        .write_as_text(requirementsString);

    _logger.debug("Runtime requirements string: %s length: %zu",
                  requirementsString.str().c_str(),
                  requirementsString.str().length());

    return requirementsString.str();
}

}  // namespace intel_npu