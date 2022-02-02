// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"
#include <gna/gna_config.hpp>
#include "gna_plugin.hpp"
#include "gna_plugin_config.hpp"
#include "common/gna_target.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_common.h"
#include <caseless.hpp>
#include <unordered_map>
#include <openvino/util/common_util.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace GNAPluginNS {
const uint8_t Config::max_num_requests;

OPENVINO_SUPPRESS_DEPRECATED_START
static const caseless_unordered_map<std::string, std::pair<Gna2AccelerationMode, bool>> supported_values = {
    {GNAConfigParams::GNA_AUTO,                 {Gna2AccelerationModeAuto,                          false}},
    {GNAConfigParams::GNA_HW,                   {Gna2AccelerationModeHardware,                      false}},
    {GNAConfigParams::GNA_HW_WITH_SW_FBACK,     {Gna2AccelerationModeHardwareWithSoftwareFallback,  false}},
    {GNAConfigParams::GNA_SW,                   {Gna2AccelerationModeSoftware,                      false}},
    {GNAConfigParams::GNA_SW_EXACT,             {Gna2AccelerationModeSoftware,                      true}},
    {GNAConfigParams::GNA_GEN,                  {Gna2AccelerationModeGeneric,                       false}},
    {GNAConfigParams::GNA_GEN_EXACT,            {Gna2AccelerationModeGeneric,                       true}},
    {GNAConfigParams::GNA_SSE,                  {Gna2AccelerationModeSse4x2,                        false}},
    {GNAConfigParams::GNA_SSE_EXACT,            {Gna2AccelerationModeSse4x2,                        true}},
    {GNAConfigParams::GNA_AVX1,                 {Gna2AccelerationModeAvx1,                          false}},
    {GNAConfigParams::GNA_AVX1_EXACT,           {Gna2AccelerationModeAvx1,                          true}},
    {GNAConfigParams::GNA_AVX2,                 {Gna2AccelerationModeAvx2,                          false}},
    {GNAConfigParams::GNA_AVX2_EXACT,           {Gna2AccelerationModeAvx2,                          true}},
};
OPENVINO_SUPPRESS_DEPRECATED_END

static const std::set<std::string> supportedTargets = {
    common::kGnaTarget2_0,
    common::kGnaTarget3_0,
    common::kGnaTarget3_5,
    common::kGnaTargetUnspecified
};

void Config::UpdateFromMap(const std::map<std::string, std::string>& config) {
    for (auto&& item : config) {
        auto key = item.first;
        auto value = item.second;

        auto fp32eq = [](float p1, float p2) -> bool {
            return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
        };

        auto check_scale_factor = [&] (float scale_factor) {
            if (fp32eq(scale_factor, 0.0f) || std::isinf(scale_factor)) {
                THROW_GNA_EXCEPTION << "input scale factor of 0.0f or +-inf not supported";
            }
        };

        auto &log = gnalog();

        auto check_compatibility = [&](const std::string& recommended_key) {
            if (config.count(recommended_key)) {
                if (value != config.at(recommended_key)) {
                    THROW_GNA_EXCEPTION << "Both " << key << " and " << recommended_key
                        << " properties are specified! Please use " << recommended_key << " only!";
                }
            }
        };

        auto get_max_num_requests = [&] () -> uint8_t {
            uint64_t num_requests;
            try {
                num_requests = std::stoul(value);
                if (num_requests == 0 || num_requests > Config::max_num_requests) {
                    throw std::out_of_range("");
                }
            } catch (std::invalid_argument&) {
                THROW_GNA_EXCEPTION << "Invalid value of number of requests/threads";
            }
            return static_cast<uint8_t>(num_requests);
        };

        auto set_target = [&](const std::string& target) {
            if (supportedTargets.count(target) == 0) {
                THROW_GNA_EXCEPTION << "Unsupported GNA config value (key, value): (" << key << ", " << value << ")";
            }
            if (key == GNA_CONFIG_KEY(EXEC_TARGET) || key == ov::intel_gna::execution_target) {
                gnaExecTarget = target;
                if (gnaCompileTarget == "")
                    gnaCompileTarget = target;
            } else {
                gnaCompileTarget = target;
            }
        };

        if (key ==  ov::intel_gna::scale_factors_per_input) {
            inputScaleFactorsPerInput = ov::util::from_string(value, ov::intel_gna::scale_factors_per_input);
            for (auto&& sf : inputScaleFactorsPerInput) {
                check_scale_factor(sf.second);
            }
        } else if (key.find(GNA_CONFIG_KEY(SCALE_FACTOR)) == 0) {
            check_compatibility(ov::intel_gna::scale_factors_per_input.name());
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
            check_scale_factor(scale_factor);
            // missing scale factors are set to be 1.0f
            if (inputScaleFactors.size() <= input_index) {
                inputScaleFactors.resize(input_index + 1, GNAPluginNS::kScaleFactorDefault);
            }
            inputScaleFactors[input_index] = InferenceEngine::CNNLayer::ie_parse_float(value);
        } else if (key == GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE) || key ==  ov::intel_gna::firmware_model_image_path) {
            dumpXNNPath = value;
OPENVINO_SUPPRESS_DEPRECATED_START
        } else if (key == GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION)) {
            dumpXNNGeneration = value;
        } else if (key == GNA_CONFIG_KEY(DEVICE_MODE) || key == ov::intel_gna::execution_mode) {
            auto procType = supported_values.find(value);
            if (procType == supported_values.end()) {
                if (value == GNA_CONFIG_VALUE(SW_FP32)) {
                    gnaFlags.sw_fp32 = true;
                } else {
                    THROW_GNA_EXCEPTION << "GNA device mode unsupported: " << value;
                }
            } else {
                gnaFlags.sw_fp32 = false;
                pluginGna2AccMode = procType->second.first;
                swExactMode = procType->second.second;
            }
OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (key == ov::intel_gna::execution_target || key == ov::intel_gna::compile_target) {
            auto target = ov::util::from_string(value, ov::intel_gna::execution_target);
            std::string target_str = "";
            if (ov::intel_gna::HWGeneration::GNA_2_0 == target) {
                target_str = common::kGnaTarget2_0;
            } else if (ov::intel_gna::HWGeneration::GNA_3_0 == target) {
                target_str = common::kGnaTarget3_0;
            } else if (ov::intel_gna::HWGeneration::GNA_3_5 == target) {
                target_str = common::kGnaTarget3_5;
            }
            set_target(target_str);
        } else if (key == GNA_CONFIG_KEY(EXEC_TARGET)) {
            check_compatibility(ov::intel_gna::execution_target.name());
            set_target(value);
        } else if (key == GNA_CONFIG_KEY(COMPILE_TARGET)) {
            check_compatibility(ov::intel_gna::compile_target.name());
            set_target(value);
        } else if (key == GNA_CONFIG_KEY(COMPACT_MODE) || key ==  ov::intel_gna::memory_reuse) {
            if (value == PluginConfigParams::YES) {
                gnaFlags.compact_mode = true;
            } else if (value == PluginConfigParams::NO) {
                gnaFlags.compact_mode = false;
            } else {
                log << "GNA compact mode should be true/false (YES/NO), but not " << value;
                THROW_GNA_EXCEPTION << "GNA compact mode should be true/false (YES/NO), but not " << value;
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
        } else if (key == ov::hint::performance_mode) {
            performance_mode = ov::util::from_string(value, ov::hint::performance_mode);
        } else if (key ==  ov::hint::inference_precision) {
            std::stringstream ss(value);
            ss >> inference_precision;
            if ((inference_precision != ov::element::i8) && (inference_precision != ov::element::i16)) {
                THROW_GNA_EXCEPTION << "Unsupported precision of GNA hardware, should be i16 or i8, but was: "
                                    << value;
            }
            gnaPrecision = (inference_precision == ov::element::i8) ? Precision::I8 : Precision::I16;
        } else if (key == GNA_CONFIG_KEY(PRECISION)) {
            check_compatibility(ov::hint::inference_precision.name());
            auto precision = Precision::FromStr(value);
            if (precision != Precision::I8 && precision != Precision::I16) {
                log << "Unsupported precision of GNA hardware, should be Int16 or Int8, but was: " << value;
                THROW_GNA_EXCEPTION << "Unsupported precision of GNA hardware, should be Int16 or Int8, but was: "
                                    << value;
            }
            gnaPrecision = precision;
        } else if (key ==  ov::intel_gna::pwl_design_algorithm) {
            gnaFlags.pwl_design_algorithm = ov::util::from_string(value, ov::intel_gna::pwl_design_algorithm);
            gnaFlags.uniformPwlDesign = (gnaFlags.pwl_design_algorithm == ov::intel_gna::PWLDesignAlgorithm::UNIFORM_DISTRIBUTION) ? true : false;
OPENVINO_SUPPRESS_DEPRECATED_START
        } else if (key == GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN)) {
            check_compatibility(ov::intel_gna::pwl_design_algorithm.name());
            // This key is deprecated and will be removed in a future release
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
        } else if (key == GNA_CONFIG_KEY(PWL_MAX_ERROR_PERCENT) || key ==  ov::intel_gna::pwl_max_error_percent) {
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
OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (key == CONFIG_KEY(PERF_COUNT) || key == ov::enable_profiling) {
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
OPENVINO_SUPPRESS_DEPRECATED_START
        } else if (key == ov::hint::num_requests) {
            try {
                gnaFlags.num_requests = get_max_num_requests();
            } catch (std::out_of_range&) {
                gnaFlags.num_requests =  (0 == stoul(value)) ? 1 : Config::max_num_requests;
            }
        } else if (key == GNA_CONFIG_KEY(LIB_N_THREADS)) {
            check_compatibility(ov::hint::num_requests.name());
            try {
                gnaFlags.num_requests = get_max_num_requests();
            } catch (std::out_of_range&) {
                log << "Unsupported accelerator lib number of threads: " << value
                    << ", should be greater than 0 and less than " << Config::max_num_requests;
                THROW_GNA_EXCEPTION << "Unsupported accelerator lib number of threads: " << value
                                    << ", should be greater than 0 and less than" << Config::max_num_requests;
            }
        } else if (key == CONFIG_KEY(SINGLE_THREAD)) {
            if (value == PluginConfigParams::YES) {
                gnaFlags.gna_openmp_multithreading = false;
            } else if (value == PluginConfigParams::NO) {
                gnaFlags.gna_openmp_multithreading = true;
            } else {
                log << "SINGLE_THREAD should be YES/NO, but not" << value;
                THROW_GNA_EXCEPTION << "SINGLE_THREAD should be YES/NO, but not" << value;
            }
OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (key == CONFIG_KEY(LOG_LEVEL) || key == ov::log::level) {
            if (value == PluginConfigParams::LOG_WARNING || value == PluginConfigParams::LOG_NONE || value == PluginConfigParams::LOG_DEBUG) {
                gnaFlags.log_level = ov::util::from_string(value, ov::log::level);
            } else {
                log << "Currently only LOG_LEVEL = LOG_WARNING, LOG_DEBUG and LOG_NONE are supported, not " << value;
                THROW_GNA_EXCEPTION << "Currently only LOG_LEVEL = LOG_WARNING, LOG_DEBUG and LOG_NONE are supported, not " << value;
            }
        } else {
            IE_THROW(NotFound)
                << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
                << "Incorrect GNA Plugin config. Key " << item.first << " not supported";
        }

        if (gnaFlags.sw_fp32 && gnaFlags.num_requests > 1) {
            THROW_GNA_EXCEPTION << "GNA plugin does not support async mode on GNA_SW_FP32!";
        }
    }

    if (inputScaleFactorsPerInput.empty() && inputScaleFactors.empty()) {
        inputScaleFactors.push_back(1.0f);
    }

    AdjustKeyMapValues();
}

void Config::AdjustKeyMapValues() {
    std::lock_guard<std::mutex> lockGuard{ mtx4keyConfigMap };
    keyConfigMap.clear();

    if (!inputScaleFactorsPerInput.empty()) {
        keyConfigMap[ov::intel_gna::scale_factors_per_input.name()] =
            ov::util::to_string(inputScaleFactorsPerInput);
    } else {
        if (inputScaleFactors.empty()) {
            inputScaleFactors.push_back(1.0);
        }
        keyConfigMap[GNA_CONFIG_KEY(SCALE_FACTOR)] = std::to_string(inputScaleFactors[0]);
        for (int n = 0; n < inputScaleFactors.size(); n++) {
            keyConfigMap[GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(n)] =
                    std::to_string(inputScaleFactors[n]);
        }
    }
    keyConfigMap[ov::intel_gna::firmware_model_image_path.name()] = dumpXNNPath;
    IE_SUPPRESS_DEPRECATED_START
    keyConfigMap[GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION)] = dumpXNNGeneration;
    IE_SUPPRESS_DEPRECATED_END
    std::string device_mode;
    if (gnaFlags.sw_fp32) {
        device_mode = ov::util::to_string(ov::intel_gna::ExecutionMode::SW_FP32);
    } else {
        for (auto&& value : supported_values) {
            if (value.second.first == pluginGna2AccMode &&
                value.second.second == swExactMode) {
                device_mode = ov::util::to_string(value.first);
                break;
            }
        }
    }
    IE_ASSERT(!device_mode.empty());
    keyConfigMap[ov::intel_gna::execution_mode.name()] = device_mode;
    keyConfigMap[GNA_CONFIG_KEY(EXEC_TARGET)] = gnaExecTarget;
    keyConfigMap[GNA_CONFIG_KEY(COMPILE_TARGET)] = gnaCompileTarget;
    keyConfigMap[ov::intel_gna::memory_reuse.name()] =
            gnaFlags.compact_mode ? PluginConfigParams::YES : PluginConfigParams::NO;
    keyConfigMap[CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)] =
            gnaFlags.exclusive_async_requests ? PluginConfigParams::YES: PluginConfigParams::NO;
    keyConfigMap[ov::hint::performance_mode.name()] = ov::util::to_string(performance_mode);
    if (inference_precision != ov::element::undefined) {
        keyConfigMap[ov::hint::inference_precision.name()] = ov::util::to_string(inference_precision);
    } else {
        keyConfigMap[GNA_CONFIG_KEY(PRECISION)] = gnaPrecision.name();
    }
OPENVINO_SUPPRESS_DEPRECATED_START
    if (gnaFlags.pwl_design_algorithm != ov::intel_gna::PWLDesignAlgorithm::UNDEFINED) {
        keyConfigMap[ov::intel_gna::pwl_design_algorithm.name()] =
                ov::util::to_string(gnaFlags.pwl_design_algorithm);
    } else {
        keyConfigMap[GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN)] =
                gnaFlags.uniformPwlDesign ? PluginConfigParams::YES: PluginConfigParams::NO;
    }
    keyConfigMap[ov::intel_gna::pwl_max_error_percent.name()] = std::to_string(gnaFlags.pwlMaxErrorPercent);
    keyConfigMap[ov::hint::num_requests.name()] = std::to_string(gnaFlags.num_requests);
    keyConfigMap[GNA_CONFIG_KEY(LIB_N_THREADS)] = std::to_string(gnaFlags.num_requests);
    keyConfigMap[CONFIG_KEY(SINGLE_THREAD)] =
            gnaFlags.gna_openmp_multithreading ? PluginConfigParams::NO: PluginConfigParams::YES;
OPENVINO_SUPPRESS_DEPRECATED_END
    keyConfigMap[ov::enable_profiling.name()] =
            gnaFlags.performance_counting ? PluginConfigParams::YES: PluginConfigParams::NO;
    keyConfigMap[ov::log::level.name()] = ov::util::to_string(gnaFlags.log_level);
}

Parameter Config::GetParameter(const std::string& name) const {
    std::lock_guard<std::mutex> lockGuard{ mtx4keyConfigMap };
    if (name == ov::intel_gna::scale_factors_per_input) {
        return decltype(ov::intel_gna::scale_factors_per_input)::value_type {inputScaleFactorsPerInput};
    } else if (name == ov::intel_gna::pwl_design_algorithm) {
        return gnaFlags.pwl_design_algorithm;
    } else if (name ==  ov::intel_gna::execution_target) {
        return ((gnaExecTarget == common::kGnaTarget2_0) ? ov::intel_gna::HWGeneration::GNA_2_0 :
                (gnaExecTarget == common::kGnaTarget3_0) ? ov::intel_gna::HWGeneration::GNA_3_0 :
                (gnaExecTarget == common::kGnaTarget3_5) ? ov::intel_gna::HWGeneration::GNA_3_5 :
                ov::intel_gna::HWGeneration::UNDEFINED);
    } else if (name ==  ov::intel_gna::compile_target) {
        return ((gnaCompileTarget == common::kGnaTarget2_0) ? ov::intel_gna::HWGeneration::GNA_2_0 :
                (gnaCompileTarget == common::kGnaTarget3_0) ? ov::intel_gna::HWGeneration::GNA_3_0 :
                (gnaCompileTarget == common::kGnaTarget3_5) ? ov::intel_gna::HWGeneration::GNA_3_5 :
                ov::intel_gna::HWGeneration::UNDEFINED);
    } else if (name == ov::hint::performance_mode) {
        return performance_mode;
    } else if (name ==  ov::hint::inference_precision) {
        return inference_precision;
    } else {
        auto result = keyConfigMap.find(name);
        if (result == keyConfigMap.end()) {
            THROW_GNA_EXCEPTION << "Unsupported config key: " << name;
        }
        return result->second;
    }
}

const Parameter Config::GetSupportedProperties(bool compiled) {
    ov::PropertyMutability model_mutability = compiled ? ov::PropertyMutability::RO : ov::PropertyMutability::RW;
    const std::vector<ov::PropertyName> supported_properties = {
        { ov::supported_properties.name(), ov::PropertyMutability::RO },
        { ov::available_devices.name(), ov::PropertyMutability::RO },
        { ov::optimal_number_of_infer_requests.name(), ov::PropertyMutability::RO },
        { ov::range_for_async_infer_requests.name(), ov::PropertyMutability::RO },
        { ov::device::capabilities.name(), ov::PropertyMutability::RO },
        { ov::device::full_name.name(), ov::PropertyMutability::RO },
        { ov::intel_gna::library_full_version.name(), ov::PropertyMutability::RO },
        { ov::intel_gna::scale_factors_per_input.name(), model_mutability },
        { ov::intel_gna::firmware_model_image_path.name(), model_mutability },
        { ov::intel_gna::execution_mode.name(), ov::PropertyMutability::RW },
        { ov::intel_gna::execution_target.name(), model_mutability },
        { ov::intel_gna::compile_target.name(), model_mutability },
        { ov::intel_gna::pwl_design_algorithm.name(), model_mutability },
        { ov::intel_gna::pwl_max_error_percent.name(), model_mutability },
        { ov::hint::performance_mode.name(), ov::PropertyMutability::RW },
        { ov::hint::inference_precision.name(), model_mutability },
        { ov::hint::num_requests.name(), model_mutability },
        { ov::log::level.name(), ov::PropertyMutability::RW },
    };
    return supported_properties;
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
