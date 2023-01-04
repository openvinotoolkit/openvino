// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale_factor_helper.hpp"

#include "common/numerical_utils.hpp"
#include "gna_plugin_config.hpp"
#include "log/log.hpp"

namespace ov {
namespace intel_gna {
using namespace common;
namespace helpers {

static bool IsCustomInputScaleFactorAvailableLegacy(const std::vector<float>& input_scale_factors) {
    if (input_scale_factors.empty()) {
        return false;
    }

    bool is_scale_factor_custom = false;
    for (const auto& scale_factor : input_scale_factors) {
        if (!AreFpEq(scale_factor, kScaleFactorDefault)) {
            is_scale_factor_custom = true;
            break;
        }
    }

    return is_scale_factor_custom;
}

static void ApplyScaleFactorsLegacy(const std::vector<float>& input_scale_factors, GnaInputs& inputs) {
    for (size_t id = 0; id < inputs.size(); ++id) {
        log::warning() << "Using input scale factor: " << input_scale_factors[id]
                         << ", defined in configuration for input id: " << id << std::endl;
        if (input_scale_factors.size() > id) {
            inputs.Get().at(id).scale_factor = input_scale_factors[id];
        } else {
            log::warning() << "Configuration input scale factors count is lower than inputs count" << std::endl;
        }
    }
}

static bool IsCustomInputScaleFactorPerInputAvailable(const std::map<std::string, float>& per_input_scale_factors) {
    if (per_input_scale_factors.empty()) {
        return false;
    }

    bool is_scale_factor_custom = false;
    for (const auto& scale_factor : per_input_scale_factors) {
        if (!AreFpEq(scale_factor.second, kScaleFactorDefault)) {
            is_scale_factor_custom = true;
            break;
        }
    }

    return is_scale_factor_custom;
}

static void ApplyScaleFactorsPerInput(const std::map<std::string, float>& per_input_scale_factors,
                                      GnaInputs& inputs) {
    if (per_input_scale_factors.size() > inputs.size()) {
        IE_THROW() << "Configuration per input scale factors count is bigger than inputs count";
    }

    for (auto&& sf : per_input_scale_factors) {
        // to support the both legacy and 2.0 API we need to check all possible names in the configuration
        auto input_it = std::find_if(inputs.Get().begin(), inputs.Get().end(), [&](const InputDesc& input_desc) {
                return sf.first == input_desc.name || input_desc.tensor_names.count(sf.first);
            });

        if (input_it == inputs.end()) {
            IE_THROW() << "Given scale factor for invalid input: " << sf.first;
        }
        ::log::warning() << "Using input scale factor: " << sf.second
                         << ", defined in configuration for input: " << sf.first << std::endl;
        input_it->scale_factor = sf.second;
    }
}

static bool CheckIfCanApplyCustomScaleFactor(const header_latest::ModelHeader& header) {
    static constexpr header_2_dot_8::ModelHeader::Version sc_forbid_override_scale_factor;
    if (!header_latest::IsFirstVersionLower(header.version, sc_forbid_override_scale_factor)) {
        ::log::warning() << "Cannot apply custom scale factor for model versions >= "
                         << sc_forbid_override_scale_factor.major << "." << sc_forbid_override_scale_factor.minor
                         << std::endl;
        return false;
    }
    return true;
}

void ApplyInputScaleFactors(GnaInputs& inputs, const Config& config, const header_latest::ModelHeader& header) {
    if (CheckIfCanApplyCustomScaleFactor(header)) {
        ApplyInputScaleFactors(inputs, config);
    }
}

void ApplyInputScaleFactors(GnaInputs& inputs, const Config& config) {
    // If scale factors are defined in configuration we still need to use them instead of imported values,
    // for example to change the scale factors for the old models.
    const bool custom_scale_factor_per_input =
        IsCustomInputScaleFactorPerInputAvailable(config.inputScaleFactorsPerInput);
    const bool custom_scale_factor_legacy = IsCustomInputScaleFactorAvailableLegacy(config.inputScaleFactors);

    if (!custom_scale_factor_per_input && !custom_scale_factor_legacy) {
        return;
    }

    // Due the fact inputScaleFactors is set by defuault construcor of ov::intel_gna::Config
    // we need to check is_intput_scale_factor_per_input_given as first.
    if (custom_scale_factor_per_input) {
        ApplyScaleFactorsPerInput(config.inputScaleFactorsPerInput, inputs);
        return;
    }

    ApplyScaleFactorsLegacy(config.inputScaleFactors, inputs);
}

}  // namespace helpers
}  // namespace intel_gna
}  // namespace ov