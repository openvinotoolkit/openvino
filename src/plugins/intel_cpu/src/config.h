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

    Config clone() const;
    Config clone(int num_sub_streamst) const;

    Config(const Config& other);
    Config& operator=(const Config& other);

    void set_properties(const ov::AnyMap& config, OptionVisibility allowed_visibility = OptionVisibility::ANY);

private:
    void finalize_impl(const IRemoteContext* context) override;
    void apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) override;
    void apply_rt_info(const IRemoteContext* context, const ov::RTMap& rt_info);
    void apply_cpu_rt_info(const ov::RTMap& rt_info);

    void apply_user_properties();
    void apply_hints();
    void set_default_values();
    void apply_execution_hints();
    void apply_performance_hints();
    void apply_threading_properties();

    int get_default_num_streams();
    std::vector<std::vector<int>> get_default_proc_type_table();

    std::vector<std::vector<int>> generate_stream_info(int streams);

    #include "config_options.inl"

public:
    int get_model_prefer_threads() const {
        return m_model_prefer_threads;
    }

    int get_num_sub_streams() const {
        return m_num_sub_streams;
    }

    const std::vector<std::vector<int>>& get_stream_rank_table() const {
        return m_stream_rank_table;
    }

    const std::vector<std::vector<int>>& get_stream_info_table() const {
        return m_stream_info_table;
    }

private:
    int m_model_prefer_threads = -1;
    std::vector<std::vector<int>> m_stream_rank_table = {};
    std::vector<std::vector<int>> m_stream_info_table = {};
    int m_num_sub_streams = 0;
    std::vector<std::vector<int>> m_proc_type_table = {};
    int m_numa_node_id = -1;

    friend class StreamsCalculationTests;
    friend class StreamGenerationTests;
};

}  // namespace ov::intel_cpu
