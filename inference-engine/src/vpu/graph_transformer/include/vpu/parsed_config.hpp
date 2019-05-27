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

#include <vpu/graph_transformer.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/logger.hpp>
#include <vpu/utils/enums.hpp>

namespace vpu {

VPU_DECLARE_ENUM(ConfigMode,
    DEFAULT_MODE = 0,
    RUNTIME_MODE = 1,
    COMPILE_MODE = 2,
)

struct ParsedConfig {
    CompilationConfig compileConfig;

    bool printReceiveTensorTime = false;
    bool exclusiveAsyncRequests = false;
    bool perfCount              = false;
    bool forceReset             = false;

    LogLevel vpuLogLevel = LogLevel::None;
    LogLevel logLevel = LogLevel::None;

    PerfReport perfReport = PerfReport::PerStage;

    std::map<std::string, std::string> getDefaultConfig() const;

    virtual ~ParsedConfig() = default;

protected:
    explicit ParsedConfig(ConfigMode configMode = ConfigMode::DEFAULT_MODE);

    void checkUnknownOptions(const std::map<std::string, std::string> &config) const;
    virtual void checkInvalidValues(const std::map<std::string, std::string> &config) const;
    std::unordered_set<std::string> getKnownOptions() const;

    std::map<std::string, std::string> parse(const std::map<std::string, std::string> &config) {
        checkInvalidValues(config);
        checkUnknownOptions(config);
        checkOptionsAccordingToMode(config);

        auto defaultConfig = getDefaultConfig();
        for (auto &&entry : config) {
            defaultConfig[entry.first] = entry.second;
        }

        return defaultConfig;
    }

    void configure(const std::map<std::string, std::string> &config);
    void checkSupportedValues(const std::unordered_map<std::string, std::unordered_set<std::string>> &supported,
                              const std::map<std::string, std::string> &config) const;

    virtual void checkOptionsAccordingToMode(const std::map<std::string, std::string> &config) const;
    virtual std::unordered_set<std::string> getCompileOptions() const;
    virtual std::unordered_set<std::string> getRuntimeOptions() const;

private:
    ConfigMode _mode = ConfigMode::DEFAULT_MODE;
    Logger::Ptr _log;
};

template<typename T, typename V>
inline void setOption(T &dst, const V &supported, const std::map<std::string, std::string> &config, const std::string &key) {
    auto value = config.find(key);
    if (value != config.end()) {
        dst = supported.at(value->second);
    }
}

inline void setOption(std::string &dst, const std::map<std::string, std::string> &config, const std::string &key) {
    auto value = config.find(key);
    if (value != config.end()) {
        dst = value->second;
    }
}

template<typename T, typename C>
inline void setOption(T &dst, const std::map<std::string, std::string> &config, const std::string &key, const C &preprocess) {
    auto value = config.find(key);
    if (value != config.end()) {
        dst = preprocess(value->second);
    }
}

}  // namespace vpu
