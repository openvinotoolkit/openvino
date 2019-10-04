// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <vpu/parsed_config_base.hpp>

#include <vpu/graph_transformer.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/enums.hpp>

namespace vpu {

struct ParsedConfig : public ParsedConfigBase{
    CompilationConfig compileConfig;

    bool printReceiveTensorTime = false;
    bool perfCount              = false;

    PerfReport perfReport = PerfReport::PerLayer;

    std::map<std::string, std::string> getDefaultConfig() const override;

    ~ParsedConfig() = default;

protected:
    explicit ParsedConfig(ConfigMode configMode);

    void checkInvalidValues(const std::map<std::string, std::string> &config) const override;

    void configure(const std::map<std::string, std::string> &config) override;

    std::unordered_set<std::string> getKnownOptions() const override;
    std::unordered_set<std::string> getCompileOptions() const override;
    std::unordered_set<std::string> getRuntimeOptions() const override;

private:
    ConfigMode _mode = ConfigMode::DEFAULT_MODE;
};
}  // namespace vpu
