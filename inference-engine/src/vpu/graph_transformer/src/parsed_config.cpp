// Copyright (C) 2018-2020 Intel Corporation
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
#include <ie_plugin_config.hpp>

#include <vpu/utils/string.hpp>

namespace vpu {

const std::unordered_set<std::string>& ParsedConfig::getCompileOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfigBase::getCompileOptions(), {
        //
        // Public options
        //

        CONFIG_KEY(CONFIG_FILE),

        VPU_CONFIG_KEY(NETWORK_CONFIG),
        VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
        VPU_CONFIG_KEY(HW_EXTRA_SPLIT),
        VPU_CONFIG_KEY(CUSTOM_LAYERS),

        VPU_CONFIG_KEY(INPUT_NORM),
        VPU_CONFIG_KEY(INPUT_BIAS),

        //
        // Private options
        //

        VPU_CONFIG_KEY(NUMBER_OF_SHAVES),
        VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES),
        VPU_CONFIG_KEY(TILING_CMX_LIMIT_KB),

        VPU_CONFIG_KEY(TENSOR_STRIDES),

        VPU_CONFIG_KEY(IR_WITH_SCALES_DIRECTORY),
        VPU_CONFIG_KEY(DETECT_NETWORK_BATCH),
        VPU_CONFIG_KEY(COPY_OPTIMIZATION),
        VPU_CONFIG_KEY(HW_INJECT_STAGES),
        VPU_CONFIG_KEY(HW_POOL_CONV_MERGE),
        VPU_CONFIG_KEY(PACK_DATA_IN_CMX),
        VPU_CONFIG_KEY(HW_DILATION),
        VPU_CONFIG_KEY(FORCE_DEPRECATED_CNN_CONVERSION),
        VPU_CONFIG_KEY(DISABLE_REORDER),
        VPU_CONFIG_KEY(ENABLE_PERMUTE_MERGING),
        VPU_CONFIG_KEY(ENABLE_REPL_WITH_SCRELU),
        VPU_CONFIG_KEY(ENABLE_REPLACE_WITH_REDUCE_MEAN),
        VPU_CONFIG_KEY(ENABLE_TENSOR_ITERATOR_UNROLLING),
        VPU_CONFIG_KEY(FORCE_PURE_TENSOR_ITERATOR),
        VPU_CONFIG_KEY(DISABLE_CONVERT_STAGES),

        //
        // Debug options
        //

        VPU_CONFIG_KEY(HW_WHITE_LIST),
        VPU_CONFIG_KEY(HW_BLACK_LIST),

        VPU_CONFIG_KEY(NONE_LAYERS),
        VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS),

        VPU_CONFIG_KEY(COMPILER_LOG_FILE_PATH),

        VPU_CONFIG_KEY(DUMP_INTERNAL_GRAPH_FILE_NAME),
        VPU_CONFIG_KEY(DUMP_INTERNAL_GRAPH_DIRECTORY),
        VPU_CONFIG_KEY(DUMP_ALL_PASSES),
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

const std::unordered_set<std::string>& ParsedConfig::getDeprecatedOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfigBase::getDeprecatedOptions(), {
        VPU_CONFIG_KEY(INPUT_NORM),
        VPU_CONFIG_KEY(INPUT_BIAS),
        VPU_CONFIG_KEY(NETWORK_CONFIG),
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

void ParsedConfig::parse(const std::map<std::string, std::string>& config) {
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

    const auto parseStringSet = [](const std::string& value) {
        return splitStringList<ie::details::caseless_set<std::string>>(value, ',');
    };

    ParsedConfigBase::parse(config);

    setOption(_compilerLogFilePath, config, VPU_CONFIG_KEY(COMPILER_LOG_FILE_PATH));
    setOption(_compileConfig.dumpInternalGraphFileName, config, VPU_CONFIG_KEY(DUMP_INTERNAL_GRAPH_FILE_NAME));
    setOption(_compileConfig.dumpInternalGraphDirectory, config, VPU_CONFIG_KEY(DUMP_INTERNAL_GRAPH_DIRECTORY));
    setOption(_compileConfig.dumpAllPasses, switches, config, VPU_CONFIG_KEY(DUMP_ALL_PASSES));

    setOption(_compileConfig.detectBatch,                    switches, config, VPU_CONFIG_KEY(DETECT_NETWORK_BATCH));
    setOption(_compileConfig.copyOptimization,               switches, config, VPU_CONFIG_KEY(COPY_OPTIMIZATION));
    setOption(_compileConfig.packDataInCmx,                  switches, config, VPU_CONFIG_KEY(PACK_DATA_IN_CMX));
    setOption(_compileConfig.ignoreUnknownLayers,            switches, config, VPU_CONFIG_KEY(IGNORE_UNKNOWN_LAYERS));
    setOption(_compileConfig.hwOptimization,                 switches, config, VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION));
    setOption(_compileConfig.hwExtraSplit,                   switches, config, VPU_CONFIG_KEY(HW_EXTRA_SPLIT));
    setOption(_compileConfig.injectSwOps,                    switches, config, VPU_CONFIG_KEY(HW_INJECT_STAGES));
    setOption(_compileConfig.mergeHwPoolToConv,              switches, config, VPU_CONFIG_KEY(HW_POOL_CONV_MERGE));
    setOption(_compileConfig.hwDilation,                     switches, config, VPU_CONFIG_KEY(HW_DILATION));
    setOption(_compileConfig.forceDeprecatedCnnConversion,   switches, config, VPU_CONFIG_KEY(FORCE_DEPRECATED_CNN_CONVERSION));
    setOption(_compileConfig.disableReorder,                 switches, config, VPU_CONFIG_KEY(DISABLE_REORDER));
    setOption(_compileConfig.enablePermuteMerging,           switches, config, VPU_CONFIG_KEY(ENABLE_PERMUTE_MERGING));
    setOption(_compileConfig.enableReplWithSCRelu,           switches, config, VPU_CONFIG_KEY(ENABLE_REPL_WITH_SCRELU));
    setOption(_compileConfig.enableReplaceWithReduceMean,    switches, config, VPU_CONFIG_KEY(ENABLE_REPLACE_WITH_REDUCE_MEAN));
    setOption(_compileConfig.enableTensorIteratorUnrolling,  switches, config, VPU_CONFIG_KEY(ENABLE_TENSOR_ITERATOR_UNROLLING));
    setOption(_compileConfig.forcePureTensorIterator,        switches, config, VPU_CONFIG_KEY(FORCE_PURE_TENSOR_ITERATOR));
    setOption(_compileConfig.disableConvertStages,           switches, config, VPU_CONFIG_KEY(DISABLE_CONVERT_STAGES));

    setOption(_compileConfig.irWithVpuScalesDir, config, VPU_CONFIG_KEY(IR_WITH_SCALES_DIRECTORY));
    setOption(_compileConfig.noneLayers,    config, VPU_CONFIG_KEY(NONE_LAYERS), parseStringSet);
    setOption(_compileConfig.hwWhiteList,   config, VPU_CONFIG_KEY(HW_WHITE_LIST), parseStringSet);
    setOption(_compileConfig.hwBlackList,   config, VPU_CONFIG_KEY(HW_BLACK_LIST), parseStringSet);

    // Priority is set to VPU configuration file over plug-in config.
    setOption(_compileConfig.customLayers, config, VPU_CONFIG_KEY(CUSTOM_LAYERS));
    if (_compileConfig.customLayers.empty()) {
        setOption(_compileConfig.customLayers, config, CONFIG_KEY(CONFIG_FILE));
    }

    auto isPositive = [](int value) {
        return value >= 0;
    };

    auto isDefaultValue = [](int value) {
        return value == -1;
    };

    auto preprocessCompileOption = [&](const std::string& src) {
        int value = parseInt(src);

        if (isPositive(value) || isDefaultValue(value)) {
            return value;
        }

        throw std::invalid_argument("Value must be positive or default(-1).");
    };

    setOption(_compileConfig.numSHAVEs, config, VPU_CONFIG_KEY(NUMBER_OF_SHAVES), preprocessCompileOption);
    setOption(_compileConfig.numCMXSlices, config, VPU_CONFIG_KEY(NUMBER_OF_CMX_SLICES), preprocessCompileOption);
    setOption(_compileConfig.numExecutors, config, VPU_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS), preprocessCompileOption);
    setOption(_compileConfig.tilingCMXLimitKB, config, VPU_CONFIG_KEY(TILING_CMX_LIMIT_KB), preprocessCompileOption);

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

#ifndef NDEBUG
    if (const auto envVar = std::getenv("IE_VPU_COMPILER_LOG_FILE_PATH")) {
        _compilerLogFilePath = envVar;
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_FILE_NAME")) {
        _compileConfig.dumpInternalGraphFileName = envVar;
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_DIRECTORY")) {
        _compileConfig.dumpInternalGraphDirectory = envVar;
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_ALL_PASSES")) {
        _compileConfig.dumpAllPasses = std::stoi(envVar) != 0;
    }
    if (const auto envVar = std::getenv("IE_VPU_NUMBER_OF_SHAVES_AND_CMX_SLICES")) {
        _compileConfig.numSHAVEs = _compileConfig.numCMXSlices = preprocessCompileOption(envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_TILING_CMX_LIMIT_KB")) {
        _compileConfig.tilingCMXLimitKB = preprocessCompileOption(envVar);
    }
#endif
}

}  // namespace vpu
