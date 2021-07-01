// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <gna/gna_config.hpp>
#include "gna_plugin.hpp"
#include "gna_plugin_config.hpp"
#include "ie_common.h"
#include <caseless.hpp>
#include <unordered_map>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace GNAPluginNS {

#if GNA_LIB_VER == 1
static const caseless_unordered_map<std::string, uint32_t> supported_values = {
        {GNAConfigParams::GNA_AUTO,     GNA_AUTO},
        {GNAConfigParams::GNA_HW,       GNA_HARDWARE},
        {GNAConfigParams::GNA_SW,       GNA_SOFTWARE},
        {GNAConfigParams::GNA_SW_EXACT, GNA_SOFTWARE & GNA_HARDWARE}
};
static const  std::vector<std::string> supported_values_on_gna2 = {
        GNAConfigParams::GNA_HW_WITH_SW_FBACK,
        GNAConfigParams::GNA_GEN,
        GNAConfigParams::GNA_GEN_EXACT,
        GNAConfigParams::GNA_SSE,
        GNAConfigParams::GNA_SSE_EXACT,
        GNAConfigParams::GNA_AVX1,
        GNAConfigParams::GNA_AVX1_EXACT,
        GNAConfigParams::GNA_AVX2,
        GNAConfigParams::GNA_AVX2_EXACT
};
#else
static const caseless_unordered_map <std::string, std::pair<Gna2AccelerationMode, bool>> supported_values = {
                {GNAConfigParams::GNA_AUTO,             {Gna2AccelerationModeAuto,                         false}},
                {GNAConfigParams::GNA_HW,               {Gna2AccelerationModeHardware,                     false}},
                {GNAConfigParams::GNA_HW_WITH_SW_FBACK, {Gna2AccelerationModeHardwareWithSoftwareFallback, false}},
                {GNAConfigParams::GNA_SW,               {Gna2AccelerationModeSoftware,                     false}},
                {GNAConfigParams::GNA_SW_EXACT,         {Gna2AccelerationModeSoftware,                     true}},
                {GNAConfigParams::GNA_GEN,              {Gna2AccelerationModeGeneric,                      false}},
                {GNAConfigParams::GNA_GEN_EXACT,        {Gna2AccelerationModeGeneric,                      true}},
                {GNAConfigParams::GNA_SSE,              {Gna2AccelerationModeSse4x2,                       false}},
                {GNAConfigParams::GNA_SSE_EXACT,        {Gna2AccelerationModeSse4x2,                       true}},
                {GNAConfigParams::GNA_AVX1,             {Gna2AccelerationModeAvx1,                         false}},
                {GNAConfigParams::GNA_AVX1_EXACT,       {Gna2AccelerationModeAvx1,                         true}},
                {GNAConfigParams::GNA_AVX2,             {Gna2AccelerationModeAvx2,                         false}},
                {GNAConfigParams::GNA_AVX2_EXACT,       {Gna2AccelerationModeAvx2,                         true}},
        };
#endif

static const std::set<std::string> supportedTargets = {
    GNAConfigParams::GNA_TARGET_2_0,
    GNAConfigParams::GNA_TARGET_3_0,
    ""
};

void Config::UpdateFromMap(const std::map<std::string, std::string>& config) {
    for (auto&& item : config) {
        auto key = item.first;
        auto value = item.second;

        auto fp32eq = [](float p1, float p2) -> bool {
            return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
        };

        auto &log = gnalog();

        if (key.find(GNA_CONFIG_KEY(SCALE_FACTOR)) == 0) {
            uint64_t input_index;
            if (key == GNA_CONFIG_KEY(SCALE_FACTOR)) {
                input_index = 0;
            } else {
                key.erase(0, strlen(GNA_CONFIG_KEY(SCALE_FACTOR)));
                if (key[0] != '_') {
                    THROW_GNA_EXCEPTION << "Invalid format of scale factor configuration key";
                }
                key.erase(0, 1);
                try {
                    input_index = std::stoi(key);
                    if (input_index > 99) {
                        throw std::out_of_range("");
                    }
                } catch (std::invalid_argument&) {
                    THROW_GNA_EXCEPTION << "Invalid value of index of input scale factor";
                } catch (std::out_of_range&) {
                    THROW_GNA_EXCEPTION << "Index of input scale factor must be in the range [0..99], " << key << " provided";
                }
            }
            auto scale_factor = InferenceEngine::CNNLayer::ie_parse_float(value);
            if (fp32eq(scale_factor, 0.0f)) {
                THROW_GNA_EXCEPTION << "input scale factor of 0.0f not supported";
            }
            // missing scale factors are set to be 1.0f
            if (inputScaleFactors.size() <= input_index) {
                inputScaleFactors.resize(input_index + 1, 1.f);
            }
            inputScaleFactors[input_index] = InferenceEngine::CNNLayer::ie_parse_float(value);
        } else if (key == GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE)) {
            dumpXNNPath = value;
        } else if (key == GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION)) {
            dumpXNNGeneration = value;
        } else if (key == GNA_CONFIG_KEY(DEVICE_MODE)) {
            auto procType = supported_values.find(value);
            if (procType == supported_values.end()) {
                if (value == GNA_CONFIG_VALUE(SW_FP32)) {
                    gnaFlags.sw_fp32 = true;
                } else {
#if GNA_LIB_VER == 1
                    auto is_gna2_mode = std::find(
                            supported_values_on_gna2.begin(),
                            supported_values_on_gna2.end(),
                            value);
                    if (is_gna2_mode != supported_values_on_gna2.end()) {
                        THROW_GNA_EXCEPTION << "This GNA device mode requires GNA2 library: " << value;
                    }
#endif
                    THROW_GNA_EXCEPTION << "GNA device mode unsupported: " << value;
                }
            } else {
#if GNA_LIB_VER == 1
                gna_proc_type = static_cast<intel_gna_proc_t>(procType->second);
#else
                pluginGna2AccMode = procType->second.first;
                swExactMode = procType->second.second;
#endif
            }
        } else if (key == GNA_CONFIG_KEY(EXEC_TARGET) || key == GNA_CONFIG_KEY(COMPILE_TARGET)) {
            if (supportedTargets.count(value) == 0) {
                THROW_GNA_EXCEPTION << "Unsupported GNA config value (key, value): (" << key << ", " << value << ")";
            }
            (key == GNA_CONFIG_KEY(EXEC_TARGET) ? gnaExecTarget : gnaCompileTarget) = value;
        } else if (key == GNA_CONFIG_KEY(COMPACT_MODE)) {
            if (value == PluginConfigParams::YES) {
                gnaFlags.compact_mode = true;
            } else if (value == PluginConfigParams::NO) {
                gnaFlags.compact_mode = false;
            } else {
                log << "GNA compact mode should be YES/NO, but not " << value;
                THROW_GNA_EXCEPTION << "GNA compact mode should be YES/NO, but not " << value;
            }
        } else if (key == CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)) {
            if (value == PluginConfigParams::YES) {
                gnaFlags.exclusive_async_requests = true;
            } else if (value == PluginConfigParams::NO) {
                gnaFlags.exclusive_async_requests = false;
            } else {
                log << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
                THROW_GNA_EXCEPTION << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
            }
        } else if (key == GNA_CONFIG_KEY(PRECISION)) {
            auto precision = Precision::FromStr(value);
            if (precision != Precision::I8 && precision != Precision::I16) {
                log << "Unsupported precision of GNA hardware, should be Int16 or Int8, but was: " << value;
                THROW_GNA_EXCEPTION << "Unsupported precision of GNA hardware, should be Int16 or Int8, but was: "
                                    << value;
            }
            gnaPrecision = precision;
        } else if (key == GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN)) {
            if (value == PluginConfigParams::YES) {
                gnaFlags.uniformPwlDesign = true;
            } else if (value == PluginConfigParams::NO) {
                gnaFlags.uniformPwlDesign = false;
            } else {
                log << "GNA pwl uniform algorithm parameter "
                    << "should be equal to YES/NO, but not" << value;
                THROW_GNA_EXCEPTION << "GNA pwl uniform algorithm parameter "
                                    << "should be equal to YES/NO, but not" << value;
            }
        } else if (key == GNA_CONFIG_KEY(PWL_MAX_ERROR_PERCENT)) {
            float max_error;
            try {
                max_error = InferenceEngine::CNNLayer::ie_parse_float(value);
                if (max_error < 0.0f || max_error > 100.0f) {
                    throw std::out_of_range("");
                }
            }
            catch (std::invalid_argument&) {
                THROW_GNA_EXCEPTION << "Invalid value of PWL max error percent";
            }
            catch (std::out_of_range&) {
                log << "Unsupported PWL error percent value: " << value
                    << ", should be greater than 0 and less than 100";
                THROW_GNA_EXCEPTION << "Unsupported PWL error percent value: " << value
                    << ", should be greater than 0 and less than 100";
            }
            gnaFlags.pwlMaxErrorPercent = max_error;
        } else if (key == CONFIG_KEY(PERF_COUNT)) {
            if (value == PluginConfigParams::YES) {
                gnaFlags.performance_counting = true;
            } else if (value == PluginConfigParams::NO) {
                gnaFlags.performance_counting = false;
            } else {
                log << "GNA performance counter enabling parameter "
                    << "should be equal to YES/NO, but not" << value;
                THROW_GNA_EXCEPTION << "GNA performance counter enabling parameter "
                                    << "should be equal to YES/NO, but not" << value;
            }
        } else if (key == GNA_CONFIG_KEY(LIB_N_THREADS)) {
            uint64_t lib_threads;
            try {
                lib_threads = std::stoul(value);
                if (lib_threads == 0 || lib_threads > (std::numeric_limits<uint8_t>::max()+1) / 2 - 1) {
                    throw std::out_of_range("");
                }
            } catch (std::invalid_argument&) {
                THROW_GNA_EXCEPTION << "Invalid value of number of threads";
            } catch (std::out_of_range&) {
                log << "Unsupported accelerator lib number of threads: " << value
                    << ", should be greater than 0 and less than 127";
                THROW_GNA_EXCEPTION << "Unsupported accelerator lib number of threads: " << value
                                    << ", should be greater than 0 and less than 127";
            }
            gnaFlags.gna_lib_async_threads_num = lib_threads;
        } else if (key == CONFIG_KEY(SINGLE_THREAD)) {
            if (value == PluginConfigParams::YES) {
                gnaFlags.gna_openmp_multithreading = false;
            } else if (value == PluginConfigParams::NO) {
                gnaFlags.gna_openmp_multithreading = true;
            } else {
                log << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
                THROW_GNA_EXCEPTION << "EXCLUSIVE_ASYNC_REQUESTS should be YES/NO, but not" << value;
            }
        } else {
            IE_THROW(NotFound)
                << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
                << "Incorrect GNA Plugin config. Key " << item.first << " not supported";
        }

        if (gnaFlags.sw_fp32 && gnaFlags.gna_lib_async_threads_num > 1) {
            THROW_GNA_EXCEPTION << "GNA plugin does not support async mode on GNA_SW_FP32!";
        }
    }

    if (inputScaleFactors.empty()) {
        inputScaleFactors.push_back(1.0f);
    }

    AdjustKeyMapValues();
}

void Config::AdjustKeyMapValues() {
    std::lock_guard<std::mutex> lockGuard{ mtx4keyConfigMap };
    keyConfigMap.clear();

    if (inputScaleFactors.empty()) {
        inputScaleFactors.push_back(1.0);
    }
    keyConfigMap[GNA_CONFIG_KEY(SCALE_FACTOR)] = std::to_string(inputScaleFactors[0]);
    for (int n = 0; n < inputScaleFactors.size(); n++) {
        keyConfigMap[GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(n)] =
                std::to_string(inputScaleFactors[n]);
    }
    keyConfigMap[GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE)] = dumpXNNPath;
    keyConfigMap[GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION)] = dumpXNNGeneration;

    std::string device_mode;
    if (gnaFlags.sw_fp32) {
        device_mode = GNA_CONFIG_VALUE(SW_FP32);
    } else {
        for (auto&& value : supported_values) {
#if GNA_LIB_VER == 1
            if (value.second == gna_proc_type) {
                device_mode = value.first;
                break;
            }
#else
            if (value.second.first == pluginGna2AccMode &&
                value.second.second == swExactMode) {
                device_mode = value.first;
                break;
            }
#endif
        }
    }
    IE_ASSERT(!device_mode.empty());
    keyConfigMap[GNA_CONFIG_KEY(DEVICE_MODE)] = device_mode;
    keyConfigMap[GNA_CONFIG_KEY(EXEC_TARGET)] = gnaExecTarget;
    keyConfigMap[GNA_CONFIG_KEY(COMPILE_TARGET)] = gnaCompileTarget;
    keyConfigMap[GNA_CONFIG_KEY(COMPACT_MODE)] =
            gnaFlags.compact_mode ? PluginConfigParams::YES: PluginConfigParams::NO;
    keyConfigMap[CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)] =
            gnaFlags.exclusive_async_requests ? PluginConfigParams::YES: PluginConfigParams::NO;
    keyConfigMap[GNA_CONFIG_KEY(PRECISION)] = gnaPrecision.name();
    keyConfigMap[GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN)] =
            gnaFlags.uniformPwlDesign ? PluginConfigParams::YES: PluginConfigParams::NO;
    keyConfigMap[GNA_CONFIG_KEY(PWL_MAX_ERROR_PERCENT)] = std::to_string(gnaFlags.pwlMaxErrorPercent);
    keyConfigMap[CONFIG_KEY(PERF_COUNT)] =
            gnaFlags.performance_counting ? PluginConfigParams::YES: PluginConfigParams::NO;
    keyConfigMap[GNA_CONFIG_KEY(LIB_N_THREADS)] = std::to_string(gnaFlags.gna_lib_async_threads_num);
    keyConfigMap[CONFIG_KEY(SINGLE_THREAD)] =
            gnaFlags.gna_openmp_multithreading ? PluginConfigParams::NO: PluginConfigParams::YES;
}

std::string Config::GetParameter(const std::string& name) const {
    std::lock_guard<std::mutex> lockGuard{ mtx4keyConfigMap };
    auto result = keyConfigMap.find(name);
    if (result == keyConfigMap.end()) {
        THROW_GNA_EXCEPTION << "Unsupported config key: " << name;
    }
    return result->second;
}

std::vector<std::string> Config::GetSupportedKeys() const {
    std::lock_guard<std::mutex> lockGuard{ mtx4keyConfigMap };
    std::vector<std::string> result;
    for (auto&& configOption : keyConfigMap) {
        result.push_back(configOption.first);
    }
    return result;
}

}  // namespace GNAPluginNS
