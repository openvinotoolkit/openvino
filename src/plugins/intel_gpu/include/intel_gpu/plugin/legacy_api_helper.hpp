// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/execution_config.hpp"
#include "ie_metric_helpers.hpp"
#include "ie_plugin_config.hpp"
#include "gpu/gpu_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_icore.hpp"

namespace ov {
namespace intel_gpu {

class LegacyAPIHelper {
public:
    static ov::AnyMap convert_legacy_properties(const std::map<std::string, std::string>& properties, bool is_new_api);
    static ov::AnyMap convert_legacy_properties(const ov::AnyMap& properties, bool is_new_api);
    static std::pair<std::string, ov::Any> convert_legacy_property(const std::pair<std::string, ov::Any>& legacy_property);
    static std::pair<std::string, ov::Any> convert_to_legacy_property(const std::pair<std::string, ov::Any>& property);
    static bool is_legacy_property(const std::pair<std::string, ov::Any>& property, bool is_new_api);
    static bool is_new_api_property(const std::pair<std::string, ov::Any>& property);
    static std::vector<std::string> get_supported_configs();
    static std::vector<std::string> get_supported_metrics();
};

}  // namespace intel_gpu
}  // namespace ov
