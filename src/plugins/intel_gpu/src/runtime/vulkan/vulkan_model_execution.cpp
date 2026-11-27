// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>

#include "intel_gpu/plugin/plugin.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::intel_gpu {

bool Plugin::supports_model_execution() {
    return false;
}

bool Plugin::supports_runtime_property(const std::string& name) {
    static const std::array supported_names{
        ov::supported_properties.name(),
        ov::available_devices.name(),
        ov::device::architecture.name(),
        ov::device::full_name.name(),
        ov::device::uuid.name(),
        ov::device::type.name(),
        ov::device::capabilities.name(),
        ov::device::id.name(),
        ov::compatibility_check.name(),
        ov::intel_gpu::runtime_type.name(),
        ov::intel_gpu::device_id.name(),
        ov::intel_gpu::device_total_mem_size.name(),
        ov::intel_gpu::device_max_alloc_mem_size.name(),
        ov::intel_gpu::memory_statistics.name(),
        ov::internal::supported_properties.name(),
        ov::internal::config_device_id.name(),
        ov::internal::device_id_map.name(),
    };
    return std::find(supported_names.begin(), supported_names.end(), name) != supported_names.end();
}

void Plugin::register_primitives() {}

void Plugin::validate_custom_layer_config(const ov::AnyMap& /*config*/) {}

// Model-capable implementations use per-device plugin state for batch metrics.
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
ov::Any Plugin::get_model_metric(const std::string& name, const ov::AnyMap& /*options*/, const cldnn::device_info& /*device_info*/) const {
    if (name == ov::compatibility_check) {
        return ov::CompatibilityCheck::UNSUPPORTED;
    }
    return {};
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& /*model*/, const ov::AnyMap& /*orig_config*/) const {
    OPENVINO_THROW("[GPU][Vulkan] Model compilation is unavailable");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& /*model*/,
                                                          const ov::AnyMap& /*orig_config*/,
                                                          const ov::SoPtr<ov::IRemoteContext>& /*context*/) const {
    OPENVINO_THROW("[GPU][Vulkan] Model compilation is unavailable");
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& /*model*/, const ov::AnyMap& /*orig_config*/) const {
    return {};
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*model*/, const ov::AnyMap& /*config*/) const {
    OPENVINO_THROW("[GPU][Vulkan] Model import is unavailable");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& /*model*/,
                                                         const ov::SoPtr<ov::IRemoteContext>& /*context*/,
                                                         const ov::AnyMap& /*orig_config*/) const {
    OPENVINO_THROW("[GPU][Vulkan] Model import is unavailable");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& /*model*/, const ov::AnyMap& /*config*/) const {
    OPENVINO_THROW("[GPU][Vulkan] Model import is unavailable");
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& /*model*/,
                                                         const ov::SoPtr<ov::IRemoteContext>& /*context*/,
                                                         const ov::AnyMap& /*config*/) const {
    OPENVINO_THROW("[GPU][Vulkan] Model import is unavailable");
}

}  // namespace ov::intel_gpu
