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

#include <debug.h>
#include <cpp_interfaces/exception2status.hpp>
#include <details/caseless.hpp>
#include <ie_plugin_config.hpp>

namespace vpu {

const std::unordered_set<std::string>& ParsedConfig::getCompileOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfigBase::getCompileOptions(), {
        //
        // Public options
        //

        CONFIG_KEY(CONFIG_FILE),

        VPU_CONFIG_KEY(COMPUTE_LAYOUT),
        VPU_CONFIG_KEY(NETWORK_CONFIG),
        VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
        VPU_CONFIG_KEY(CUSTOM_LAYERS),
        VPU_CONFIG_KEY(IGNORE_IR_STATISTIC),

        VPU_CONFIG_KEY(INPUT_NORM),
        VPU_CONFIG_KEY(INPUT_BIAS),

        //
        // Private options
        //

        VPU_CONFIG_KEY(NUMBER_OF_SHAVES),
        VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES),

        VPU_CONFIG_KEY(TENSOR_STRIDES),

        VPU_CONFIG_KEY(HW_ADAPTIVE_MODE),

        VPU_CONFIG_KEY(DETECT_NETWORK_BATCH),
        VPU_CONFIG_KEY(COPY_OPTIMIZATION),
        VPU_CONFIG_KEY(HW_INJECT_STAGES),
        VPU_CONFIG_KEY(HW_POOL_CONV_MERGE),
        VPU_CONFIG_KEY(PACK_DATA_IN_CMX),
        VPU_CONFIG_KEY(HW_DILATION),

        //
        // Debug options
        //

        VPU_CONFIG_KEY(HW_WHITE_LIST),
        VPU_CONFIG_KEY(HW_BLACK_LIST),

        VPU_CONFIG_KEY(NONE_LAYERS),
        VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS),
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& ParsedConfig::getRunTimeOptions() const {
    static const std::unordered_set<std::string> options = merge(ParsedConfigBase::getRunTimeOptions(), {
        CONFIG_KEY(PERF_COUNT),
        VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME),
        VPU_CONFIG_KEY(PERF_REPORT_MODE),
    });

    return options;
}

void ParsedConfig::parse(const std::map<std::string, std::string>& config) {
    static const std::unordered_map<std::string, ComputeLayout> layouts {
        { VPU_CONFIG_VALUE(AUTO), ComputeLayout::AUTO },
        { VPU_CONFIG_VALUE(NCHW), ComputeLayout::NCHW },
        { VPU_CONFIG_VALUE(NHWC), ComputeLayout::NHWC },
        { VPU_CONFIG_VALUE(NCDHW), ComputeLayout::NCDHW },
        { VPU_CONFIG_VALUE(NDHWC), ComputeLayout::NDHWC }
    };

    static const std::unordered_map<std::string, PerfReport> perfReports {
        { VPU_CONFIG_VALUE(PER_LAYER), PerfReport::PerLayer },
        { VPU_CONFIG_VALUE(PER_STAGE), PerfReport::PerStage },
    };

    static const auto parseStrides = [](const std::string& src) {
        auto configStrides = src;
        configStrides.pop_back();

        const auto inputs = ie::details::split(configStrides, "],");

        std::map<std::string, std::vector<int>> stridesMap;

        for (const auto& input : inputs) {
            std::vector<int> strides;

            const auto pair = ie::details::split(input, "[");
            IE_ASSERT(pair.size() == 2)
                    << "Invalid config value \"" << input << "\" "
                    << "for VPU_TENSOR_STRIDES, does not match the pattern: tensor_name[strides]";

            const auto strideValues = ie::details::split(pair.at(1), ",");

            for (const auto& stride : strideValues) {
                strides.insert(strides.begin(), std::stoi(stride));
            }

            stridesMap.insert({pair.at(0), strides});
        }

        return stridesMap;
    };

    ParsedConfigBase::parse(config);

    setOption(_compileConfig.forceLayout, layouts, config, VPU_CONFIG_KEY(COMPUTE_LAYOUT));

    setOption(_compileConfig.detectBatch,         switches, config, VPU_CONFIG_KEY(DETECT_NETWORK_BATCH));
    setOption(_compileConfig.copyOptimization,    switches, config, VPU_CONFIG_KEY(COPY_OPTIMIZATION));
    setOption(_compileConfig.packDataInCmx,       switches, config, VPU_CONFIG_KEY(PACK_DATA_IN_CMX));
    setOption(_compileConfig.ignoreUnknownLayers, switches, config, VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS));
    setOption(_compileConfig.hwOptimization,      switches, config, VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION));
    setOption(_compileConfig.hwAdaptiveMode,      switches, config, VPU_CONFIG_KEY(HW_ADAPTIVE_MODE));
    setOption(_compileConfig.injectSwOps,         switches, config, VPU_CONFIG_KEY(HW_INJECT_STAGES));
    setOption(_compileConfig.mergeHwPoolToConv,   switches, config, VPU_CONFIG_KEY(HW_POOL_CONV_MERGE));
    setOption(_compileConfig.ignoreIRStatistic,   switches, config, VPU_CONFIG_KEY(IGNORE_IR_STATISTIC));
    setOption(_compileConfig.hwDilation,          switches, config, VPU_CONFIG_KEY(HW_DILATION));

    setOption(_compileConfig.noneLayers,    config, VPU_CONFIG_KEY(NONE_LAYERS));
    setOption(_compileConfig.hwWhiteList,   config, VPU_CONFIG_KEY(HW_WHITE_LIST));
    setOption(_compileConfig.hwBlackList,   config, VPU_CONFIG_KEY(HW_BLACK_LIST));
    setOption(_compileConfig.networkConfig, config, VPU_CONFIG_KEY(NETWORK_CONFIG));

    // Priority is set to VPU configuration file over plug-in config.
    setOption(_compileConfig.customLayers, config, VPU_CONFIG_KEY(CUSTOM_LAYERS));
    if (_compileConfig.customLayers.empty()) {
        setOption(_compileConfig.customLayers, config, CONFIG_KEY(CONFIG_FILE));
    }

    setOption(_compileConfig.numSHAVEs, config, VPU_CONFIG_KEY(NUMBER_OF_SHAVES), parseInt);
    setOption(_compileConfig.numCMXSlices, config, VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES), parseInt);

    if ((_compileConfig.numSHAVEs < 0 && _compileConfig.numCMXSlices >= 0) ||
        (_compileConfig.numSHAVEs >= 0 && _compileConfig.numCMXSlices < 0)) {
        THROW_IE_EXCEPTION << "You should set both option for resource management: VPU_NUMBER_OF_CMX_SLICES and VPU_NUMBER_OF_SHAVES";
    }

    setOption(_compileConfig.ioStrides, config, VPU_CONFIG_KEY(TENSOR_STRIDES), parseStrides);

    setOption(_printReceiveTensorTime, switches,    config, VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME));
    setOption(_perfCount,              switches,    config, CONFIG_KEY(PERF_COUNT));
    setOption(_perfReport,             perfReports, config, VPU_CONFIG_KEY(PERF_REPORT_MODE));

IE_SUPPRESS_DEPRECATED_START
    setOption(_compileConfig.inputScale, config, VPU_CONFIG_KEY(INPUT_NORM), parseFloatReverse);
    setOption(_compileConfig.inputBias, config, VPU_CONFIG_KEY(INPUT_BIAS), parseFloat);
IE_SUPPRESS_DEPRECATED_END
}

}  // namespace vpu
