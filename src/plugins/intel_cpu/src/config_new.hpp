// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include <bitset>
// #include <map>
// #include <mutex>

// #include "openvino/core/type/element_type.hpp"
// #include "openvino/runtime/properties.hpp"
// #include "openvino/runtime/threading/istreams_executor.hpp"
// #include "openvino/util/common_util.hpp"
// #include "utils/debug_caps_config.h"

#include "openvino/runtime/plugin_config.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "internal_properties.hpp"

#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

struct ExecutionConfig : public ov::PluginConfig {
    ExecutionConfig();
    ExecutionConfig(std::initializer_list<ov::AnyMap::value_type> values) : ExecutionConfig() { set_property(ov::AnyMap(values)); }
    explicit ExecutionConfig(const ov::AnyMap& properties) : ExecutionConfig() { set_property(properties); }
    explicit ExecutionConfig(const ov::AnyMap::value_type& property) : ExecutionConfig() { set_property(property); }

    ExecutionConfig(const ExecutionConfig& other);
    ExecutionConfig& operator=(const ExecutionConfig& other);

    #define OV_CONFIG_OPTION(...) OV_CONFIG_DECLARE_GETTERS(__VA_ARGS__)
    #include "options.inl"
    #undef OV_CONFIG_OPTION

    void finalize_impl(std::shared_ptr<IRemoteContext> context) override;
    void apply_rt_info(std::shared_ptr<IRemoteContext> context, const ov::RTMap& rt_info) override;

    // TODO: move to GraphContext
    ov::threading::IStreamsExecutor::Config streamExecutorConfig;
    // TODO: make local for streams calculation logic
    int modelPreferThreads = -1;
    // TODO: move to GraphContext
    enum class ModelType { CNN, LLM, Unknown };
    ModelType modelType = ModelType::Unknown;

    bool DAZOn = false;

    std::vector<std::vector<int>> streamsRankTable;
    int streamsRankLevel = 1;
    int numSubStreams = 0;
    bool enableNodeSplit = false;

private:
    void set_default_values();
    void apply_user_properties();
    void apply_hints();
    void apply_execution_hints();
    void apply_performance_hints();
    const ov::PluginConfig::OptionsDesc& get_options_desc() const override;

    #define OV_CONFIG_OPTION(...) OV_CONFIG_DECLARE_OPTION(__VA_ARGS__)
    #include "options.inl"
    #undef OV_CONFIG_OPTION
};

}  // namespace intel_cpu
}  // namespace ov
