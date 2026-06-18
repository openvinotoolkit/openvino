// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "property_registration.hpp"

namespace intel_npu {

class CompiledModelPropertyManager final {
public:
    CompiledModelPropertyManager(const FilteredConfig& config,
                                 const std::shared_ptr<IGraph>& graph,
                                 const std::optional<int64_t>& batchSize,
                                 Logger& logger);

    void setProperty(const ov::AnyMap& properties);
    ov::Any getProperty(const std::string& name) const;

    const FilteredConfig& getConfig() const {
        return _config;
    }

private:
    void registerProperties();

    FilteredConfig _config;

    std::shared_ptr<IGraph> _graph;
    std::optional<int64_t> _batchSize;
    Logger& _logger;

    std::map<std::string, PropertyDescriptor> _properties;
    std::vector<ov::PropertyName> _supportedProperties;

    mutable std::mutex _mutex;
};

}  // namespace intel_npu
