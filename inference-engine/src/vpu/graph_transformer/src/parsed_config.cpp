// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/parsed_config.hpp>

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
    for (auto &&option : options) {
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

ParsedConfig::ParsedConfig(ConfigMode configMode): _mode(configMode) {
    _log = std::make_shared<Logger>("Config", LogLevel::Warning, consoleOutput());
}

void ParsedConfig::checkSupportedValues(
    const std::unordered_map<std::string, std::unordered_set<std::string>> &supported,
    const std::map<std::string, std::string> &config) const {

    auto contains = [](const std::unordered_set<std::string> &supported, const std::string &option) {
        return supported.find(option) != supported.end();
    };

    check_input(config, supported, contains);
}

void ParsedConfig::checkInvalidValues(const std::map<std::string, std::string> &config) const {
    const std::unordered_map<std::string, std::unordered_set<std::string>> supported_values = {
        { CONFIG_KEY(LOG_LEVEL),
          { CONFIG_VALUE(LOG_NONE), CONFIG_VALUE(LOG_WARNING), CONFIG_VALUE(LOG_INFO), CONFIG_VALUE(LOG_DEBUG) }},
        { VPU_CONFIG_KEY(LOG_LEVEL),
          { CONFIG_VALUE(LOG_NONE), CONFIG_VALUE(LOG_WARNING), CONFIG_VALUE(LOG_INFO), CONFIG_VALUE(LOG_DEBUG) }},
        { VPU_CONFIG_KEY(COMPUTE_LAYOUT),
            { VPU_CONFIG_VALUE(AUTO), VPU_CONFIG_VALUE(NCHW), VPU_CONFIG_VALUE(NHWC) }},
        { VPU_CONFIG_KEY(COPY_OPTIMIZATION),      { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(PACK_DATA_IN_CMX),      { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS),  { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { CONFIG_KEY(PERF_COUNT),                 { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),   { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(HW_ADAPTIVE_MODE),       { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(ALLOW_FP32_MODELS),      { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(HW_INJECT_STAGES),       { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(HW_POOL_CONV_MERGE),     { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(PERF_REPORT_MODE),
            { VPU_CONFIG_VALUE(PER_LAYER), VPU_CONFIG_VALUE(PER_STAGE) }},
        { VPU_CONFIG_KEY(IGNORE_IR_STATISTIC),    { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
        { VPU_CONFIG_KEY(HW_DILATION),            { CONFIG_VALUE(YES), CONFIG_VALUE(NO) }},
    };

    checkSupportedValues(supported_values, config);

IE_SUPPRESS_DEPRECATED_START
    auto config_norm = config.find(VPU_CONFIG_KEY(INPUT_NORM));
    if (config_norm != config.end()) {
        std::map<std::string, float> configFloat = {{VPU_CONFIG_KEY(INPUT_NORM), std::stof(config_norm->second)}};

        const std::unordered_map<std::string, std::unordered_set<float>> unsupported_values = {
            { VPU_CONFIG_KEY(INPUT_NORM), { 0.0f } }
        };

        auto doesNotContain = [](const std::unordered_set<float> &unsupported, float option) {
            return unsupported.find(option) == unsupported.end();
        };
        check_input(configFloat, unsupported_values, doesNotContain);
    }
IE_SUPPRESS_DEPRECATED_END

    auto number_of_shaves = config.find(VPU_CONFIG_KEY(NUMBER_OF_SHAVES));
    auto number_of_CMX = config.find(VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES));

    if (number_of_shaves != config.end()) {
        try {
            std::stoi(number_of_shaves->second);
        }
        catch(...) {
            THROW_IE_EXCEPTION << "Invalid config value for VPU_NUMBER_OF_SHAVES, can't cast to unsigned int";
        }
    }

    if (number_of_CMX != config.end()) {
        try {
            std::stoi(number_of_CMX->second);
        }
        catch(...) {
            THROW_IE_EXCEPTION << "Invalid config value for VPU_NUMBER_OF_CMX_SLICES, can't cast to unsigned int";
        }
    }

    if ((number_of_shaves != config.end()) && (number_of_CMX == config.end())) {
        THROW_IE_EXCEPTION << "You should set both option for resource management: VPU_NUMBER_OF_CMX_SLICES and VPU_NUMBER_OF_SHAVES";
    }

    if ((number_of_shaves == config.end()) && (number_of_CMX != config.end())) {
        THROW_IE_EXCEPTION << "You should set both option for resource management: VPU_NUMBER_OF_CMX_SLICES and VPU_NUMBER_OF_SHAVES";
    }
}

void ParsedConfig::checkUnknownOptions(const std::map<std::string, std::string> &config) const {
    auto knownOptions = getKnownOptions();
    for (auto &&entry : config) {
        if (knownOptions.find(entry.first) == knownOptions.end()) {
            THROW_IE_EXCEPTION << NOT_FOUND_str << entry.first << " key is not supported for VPU";
        }
    }
}

void ParsedConfig::checkOptionsAccordingToMode(const std::map<std::string, std::string> &config) const {
    auto compileOptions = getCompileOptions();
    for (auto &&entry : config) {
        std::stringstream errorMsgStream;
        if (compileOptions.find(entry.first) != compileOptions.end() && _mode == ConfigMode::RUNTIME_MODE) {
            _log->warning("%s option will be ignored. Seems you are using compiled graph", entry.first);
        }
    }
}

std::unordered_set<std::string> ParsedConfig::getCompileOptions() const {
IE_SUPPRESS_DEPRECATED_START
    return {
        VPU_CONFIG_KEY(COMPUTE_LAYOUT),
        VPU_CONFIG_KEY(NETWORK_CONFIG),
        VPU_CONFIG_KEY(HW_ADAPTIVE_MODE),
        VPU_CONFIG_KEY(ALLOW_FP32_MODELS),
        VPU_CONFIG_KEY(COPY_OPTIMIZATION),
        VPU_CONFIG_KEY(PACK_DATA_IN_CMX),
        VPU_CONFIG_KEY(DETECT_NETWORK_BATCH),
        VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS),
        VPU_CONFIG_KEY(NONE_LAYERS),
        VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
        VPU_CONFIG_KEY(HW_WHITE_LIST),
        VPU_CONFIG_KEY(HW_BLACK_LIST),
        VPU_CONFIG_KEY(CUSTOM_LAYERS),
        VPU_CONFIG_KEY(NUMBER_OF_SHAVES),
        VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES),
        VPU_CONFIG_KEY(HW_INJECT_STAGES),
        VPU_CONFIG_KEY(HW_POOL_CONV_MERGE),
        VPU_CONFIG_KEY(IGNORE_IR_STATISTIC),
        VPU_CONFIG_KEY(HW_DILATION),

        VPU_CONFIG_KEY(INPUT_NORM),
        VPU_CONFIG_KEY(INPUT_BIAS),
    };
IE_SUPPRESS_DEPRECATED_END
}

std::unordered_set<std::string> ParsedConfig::getRuntimeOptions() const {
    return {
        CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS),
        CONFIG_KEY(LOG_LEVEL),
        VPU_CONFIG_KEY(LOG_LEVEL),
        CONFIG_KEY(PERF_COUNT),
        VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME),
        CONFIG_KEY(CONFIG_FILE),
        VPU_CONFIG_KEY(PERF_REPORT_MODE),
    };
}

std::unordered_set<std::string> ParsedConfig::getKnownOptions() const {
    std::unordered_set<std::string> knownOptions;
    auto compileOptions = getCompileOptions();
    knownOptions.insert(compileOptions.begin(), compileOptions.end());

    auto runtimeOptions = getRuntimeOptions();
    knownOptions.insert(runtimeOptions.begin(), runtimeOptions.end());

    return knownOptions;
}

std::map<std::string, std::string> ParsedConfig::getDefaultConfig() const {
    return {};
}

void ParsedConfig::configure(const std::map<std::string, std::string> &config) {
    static const std::unordered_map<std::string, ComputeLayout> layouts {
        { VPU_CONFIG_VALUE(AUTO), ComputeLayout::AUTO },
        { VPU_CONFIG_VALUE(NCHW), ComputeLayout::NCHW },
        { VPU_CONFIG_VALUE(NHWC), ComputeLayout::NHWC },
    };

    setOption(compileConfig.forceLayout, layouts, config, VPU_CONFIG_KEY(COMPUTE_LAYOUT));

    static const std::unordered_map<std::string, bool> switches = {
        { CONFIG_VALUE(YES), true },
        { CONFIG_VALUE(NO), false }
    };

    setOption(compileConfig.detectBatch,         switches, config, VPU_CONFIG_KEY(DETECT_NETWORK_BATCH));
    setOption(compileConfig.copyOptimization,    switches, config, VPU_CONFIG_KEY(COPY_OPTIMIZATION));
    setOption(compileConfig.packDataInCmx,       switches, config, VPU_CONFIG_KEY(PACK_DATA_IN_CMX));
    setOption(compileConfig.ignoreUnknownLayers, switches, config, VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS));
    setOption(compileConfig.hwOptimization,      switches, config, VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION));
    setOption(compileConfig.hwAdaptiveMode,      switches, config, VPU_CONFIG_KEY(HW_ADAPTIVE_MODE));
    setOption(compileConfig.allowFP32Models,     switches, config, VPU_CONFIG_KEY(ALLOW_FP32_MODELS));
    setOption(compileConfig.injectSwOps,         switches, config, VPU_CONFIG_KEY(HW_INJECT_STAGES));
    setOption(compileConfig.mergeHwPoolToConv,   switches, config, VPU_CONFIG_KEY(HW_POOL_CONV_MERGE));
    setOption(compileConfig.ignoreIRStatistic,   switches, config, VPU_CONFIG_KEY(IGNORE_IR_STATISTIC));
    setOption(compileConfig.hwDilation,          switches, config, VPU_CONFIG_KEY(HW_DILATION));

    setOption(compileConfig.noneLayers,    config, VPU_CONFIG_KEY(NONE_LAYERS));
    setOption(compileConfig.hwWhiteList,   config, VPU_CONFIG_KEY(HW_WHITE_LIST));
    setOption(compileConfig.hwBlackList,   config, VPU_CONFIG_KEY(HW_BLACK_LIST));
    setOption(compileConfig.networkConfig, config, VPU_CONFIG_KEY(NETWORK_CONFIG));

    /* priority is set to VPU configuration file over plug-in config */
    setOption(compileConfig.customLayers, config, VPU_CONFIG_KEY(CUSTOM_LAYERS));
    if (compileConfig.customLayers.empty()) {
        setOption(compileConfig.customLayers, config, CONFIG_KEY(CONFIG_FILE));
    }

    setOption(compileConfig.numSHAVEs, config, VPU_CONFIG_KEY(NUMBER_OF_SHAVES),
              [](const std::string &src) { return std::stoi(src); });

    setOption(compileConfig.numCMXSlices, config, VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES),
              [](const std::string &src) { return std::stoi(src); });

    setOption(exclusiveAsyncRequests, switches, config, CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS));
    setOption(printReceiveTensorTime, switches, config, VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME));
    setOption(perfCount,              switches, config, CONFIG_KEY(PERF_COUNT));

    static const std::unordered_map<std::string, LogLevel> logLevels = {
        { CONFIG_VALUE(LOG_NONE), LogLevel::None },
        { CONFIG_VALUE(LOG_WARNING), LogLevel::Warning },
        { CONFIG_VALUE(LOG_INFO), LogLevel::Info },
        { CONFIG_VALUE(LOG_DEBUG), LogLevel::Debug }
    };

    setOption(hostLogLevel,   logLevels, config, CONFIG_KEY(LOG_LEVEL));
    setOption(deviceLogLevel, logLevels, config, VPU_CONFIG_KEY(LOG_LEVEL));

    static const std::unordered_map<std::string, PerfReport> perfReports {
        { VPU_CONFIG_VALUE(PER_LAYER), PerfReport::PerLayer },
        { VPU_CONFIG_VALUE(PER_STAGE), PerfReport::PerStage },
    };

    setOption(perfReport, perfReports, config, VPU_CONFIG_KEY(PERF_REPORT_MODE));

IE_SUPPRESS_DEPRECATED_START
    setOption(compileConfig.inputScale, config, VPU_CONFIG_KEY(INPUT_NORM),
              [](const std::string &src) { return 1.f / std::stof(src); });

    setOption(compileConfig.inputBias, config, VPU_CONFIG_KEY(INPUT_BIAS),
              [](const std::string &src) { return std::stof(src); });
IE_SUPPRESS_DEPRECATED_END

#ifndef NDEBUG
    if (auto envVar = std::getenv("IE_VPU_LOG_LEVEL")) {
        hostLogLevel = logLevels.at(envVar);
    }
#endif
}

}  // namespace vpu
