// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/parsed_config_base.hpp>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <memory>
#include <map>

#include <cpp_interfaces/exception2status.hpp>
#include <details/caseless.hpp>
#include <ie_plugin_config.hpp>

namespace vpu {
namespace  {
template<typename I, typename T, typename C>
void check_input(const I &input, const T &options, const C &check) {
    for (const auto& option : options) {
        auto input_entry = input.find(option.first);
        if (input_entry == input.end()) {
            continue;
        }

        auto input_key = input_entry->first;
        auto input_val = input_entry->second;
        auto values = option.second;

        if (!check(values, input_val)) {
            THROW_IE_EXCEPTION << "Incorrect value " << "\"" << input_val << "\"" << " for key " << input_key;
        }
    }
}

}  // namespace

ParsedConfigBase::ParsedConfigBase(ConfigMode configMode): _mode(configMode) {
        _log = std::make_shared<Logger>("Config", LogLevel::Warning, consoleOutput());
}

void ParsedConfigBase::checkSupportedValues(
        const std::unordered_map<std::string, std::unordered_set<std::string>> &supported,
        const std::map<std::string, std::string> &config) const {
    auto contains = [](const std::unordered_set<std::string> &supported, const std::string &option) {
        return supported.find(option) != supported.end();
    };

    check_input(config, supported, contains);
}

void ParsedConfigBase::checkUnknownOptions(const std::map<std::string, std::string> &config) const {
    auto knownOptions = getKnownOptions();
    for (auto &&entry : config) {
        if (knownOptions.find(entry.first) == knownOptions.end()) {
            THROW_IE_EXCEPTION << NOT_FOUND_str << entry.first << " key is not supported for VPU";
        }
    }
}

void ParsedConfigBase::checkOptionsAccordingToMode(const std::map<std::string, std::string> &config) const {
    auto compileOptions = getCompileOptions();
    for (auto &&entry : config) {
        std::stringstream errorMsgStream;
        if (compileOptions.find(entry.first) != compileOptions.end() && _mode == ConfigMode::RUNTIME_MODE) {
            _log->warning("%s option will be ignored. Seems you are using compiled graph", entry.first);
        }
    }
}

void ParsedConfigBase::checkInvalidValues(const std::map<std::string, std::string> &config) const {
    const std::unordered_map<std::string, std::unordered_set<std::string>> supported_values = {
        { CONFIG_KEY(LOG_LEVEL),
          { CONFIG_VALUE(LOG_NONE), CONFIG_VALUE(LOG_WARNING), CONFIG_VALUE(LOG_INFO), CONFIG_VALUE(LOG_DEBUG) }},
        { VPU_CONFIG_KEY(LOG_LEVEL),
          { CONFIG_VALUE(LOG_NONE), CONFIG_VALUE(LOG_WARNING), CONFIG_VALUE(LOG_INFO), CONFIG_VALUE(LOG_DEBUG) }},
        { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),   { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }}
    };

    checkSupportedValues(supported_values, config);
}

void ParsedConfigBase::configure(const std::map<std::string, std::string> &config) {
    static const std::unordered_map<std::string, LogLevel> logLevels = {
        { CONFIG_VALUE(LOG_NONE), LogLevel::None },
        { CONFIG_VALUE(LOG_WARNING), LogLevel::Warning },
        { CONFIG_VALUE(LOG_INFO), LogLevel::Info },
        { CONFIG_VALUE(LOG_DEBUG), LogLevel::Debug }
    };

    setOption(hostLogLevel,   logLevels, config, CONFIG_KEY(LOG_LEVEL));
    setOption(deviceLogLevel, logLevels, config, VPU_CONFIG_KEY(LOG_LEVEL));

#ifndef NDEBUG
    if (auto envVar = std::getenv("IE_VPU_LOG_LEVEL")) {
        hostLogLevel = logLevels.at(envVar);
    }
#endif

    static const std::unordered_map<std::string, bool> switches = {
            { CONFIG_VALUE(YES), true },
            { CONFIG_VALUE(NO), false }
    };

    setOption(exclusiveAsyncRequests, switches, config, CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));
}

std::unordered_set<std::string> ParsedConfigBase::getRuntimeOptions() const {
        return { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
                 CONFIG_KEY(LOG_LEVEL),
                 VPU_CONFIG_KEY(LOG_LEVEL)}; }

std::unordered_set<std::string> ParsedConfigBase::getKnownOptions() const {
    std::unordered_set<std::string> knownOptions;
    auto compileOptions = getCompileOptions();
    knownOptions.insert(compileOptions.begin(), compileOptions.end());

    auto runtimeOptions = getRuntimeOptions();
    knownOptions.insert(runtimeOptions.begin(), runtimeOptions.end());

    return knownOptions;
}
}  // namespace vpu
