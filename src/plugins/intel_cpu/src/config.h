// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/plugin_config.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "internal_properties.hpp"

#include "utils/general_utils.h"

namespace ov::intel_cpu {

struct Config : public ov::PluginConfig {
    Config();
    Config(std::initializer_list<ov::AnyMap::value_type> values) : Config() { set_property(ov::AnyMap(values)); }
    explicit Config(const ov::AnyMap& properties) : Config() { set_property(properties); }
    explicit Config(const ov::AnyMap::value_type& property) : Config() { set_property(property); }

    Config(const Config& other);
    Config& operator=(const Config& other);

    // TODO: move to GraphContext
    ov::threading::IStreamsExecutor::Config streamExecutorConfig;

    std::vector<std::vector<int>> streamsRankTable;
    int streamsRankLevel = 1;
    int numSubStreams = 0;
    bool enableNodeSplit = false;

private:
    void finalize_impl(const IRemoteContext* context) override;
    void apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) override;
    void apply_rt_info(const IRemoteContext* context, const ov::RTMap& rt_info);

    void apply_user_properties();
    void apply_hints();
    void set_default_values();
    void apply_execution_hints();
    void apply_performance_hints();

    #include "options.inl"
};

}  // namespace ov::intel_cpu
