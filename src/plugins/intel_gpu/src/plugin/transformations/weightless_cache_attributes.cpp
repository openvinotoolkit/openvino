#include "weightless_cache_attributes.hpp"

#include <cstring>

#include "intel_gpu/runtime/itt.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/weight_sharing_util.hpp"

namespace ov::intel_gpu {

MarkWeightlessCacheAttributes::MarkWeightlessCacheAttributes(std::shared_ptr<WeightlessConstantMap> original_constants)
    : m_original_constants(std::move(original_constants)) {
    OPENVINO_ASSERT(m_original_constants, "Missing original constant map for weightless cache markup");
}

bool MarkWeightlessCacheAttributes::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(MarkWeightlessCacheAttributes);
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "MarkWeightlessCacheAttributes");
    m_original_constants->clear();
    bool modified = false;
    const auto& weightless_attr = ov::WeightlessCacheAttribute::get_type_info_static();
    for (const auto& op : model->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || constant->get_rt_info().count(weightless_attr) != 0u)
            continue;
        auto source_buffer = ov::weight_sharing::Extension::get_constant_source_buffer(*constant);
        if (source_buffer) {
            const auto offset = ov::weight_sharing::Extension::get_constant_id(*constant);
            constant->get_rt_info()[weightless_attr] =
                ov::WeightlessCacheAttribute(constant->get_byte_size(), offset, constant->get_element_type());
            modified = true;
        }
    }

    for (const auto& op : model->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant)
            continue;
        auto& runtime_info = constant->get_rt_info();
        const auto attribute_it = runtime_info.find(weightless_attr);
        if (attribute_it == runtime_info.end())
            continue;
        const auto& attribute = attribute_it->second.as<ov::WeightlessCacheAttribute>();
        m_original_constants->emplace(attribute.bin_offset, std::make_pair(constant, attribute));
        const auto offset = static_cast<int64_t>(attribute.bin_offset);
        const auto offset_it = runtime_info.find("otd_bin_offset");
        if (offset_it == runtime_info.end() || offset_it->second.as<int64_t>() != offset) {
            runtime_info["otd_bin_offset"] = offset;
            modified = true;
        }
    }
    return modified;
}

RestoreWeightlessCacheAttributes::RestoreWeightlessCacheAttributes(std::shared_ptr<const WeightlessConstantMap> original_constants)
    : m_original_constants(std::move(original_constants)) {
    OPENVINO_ASSERT(m_original_constants, "Missing original constant map for weightless cache restoration");
}

bool RestoreWeightlessCacheAttributes::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RestoreWeightlessCacheAttributes);
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "RestoreWeightlessCacheAttributes");
    bool modified = false;
    const auto& weightless_attr = ov::WeightlessCacheAttribute::get_type_info_static();
    for (const auto& op : model->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || constant->get_rt_info().count(weightless_attr) != 0u)
            continue;
        const auto offset_it = constant->get_rt_info().find("otd_bin_offset");
        if (offset_it == constant->get_rt_info().end())
            continue;
        const auto offset = static_cast<size_t>(offset_it->second.as<int64_t>());
        const auto original_it = m_original_constants->find(offset);
        if (original_it == m_original_constants->end())
            continue;
        const auto& original = original_it->second.first;
        const auto byte_size = constant->get_byte_size();
        if (constant->get_element_type() != original->get_element_type() ||
            byte_size != original->get_byte_size() ||
            (byte_size != 0 && constant->get_data_ptr() != original->get_data_ptr() &&
             std::memcmp(constant->get_data_ptr(), original->get_data_ptr(), byte_size) != 0)) {
            continue;
        }
        constant->get_rt_info()[weightless_attr] = original_it->second.second;
        modified = true;
    }
    return modified;
}

}  // namespace ov::intel_gpu