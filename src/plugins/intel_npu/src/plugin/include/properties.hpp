// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "metrics.hpp"

namespace intel_npu {

enum class PropertiesType { PLUGIN, COMPILED_MODEL };

class Properties final {
public:
    /**
     * @brief Properties handler constructor
     * @param pType - type of object this handler gets attached to: PLUGIN or COMPILED_MODEL
     * @param config - reference to the global configuration table of the parent object
     * @param metrics - reference ptr to the metrics object of the parent object (PLUGIN only)
     */
    Properties(const PropertiesType pType,
               FilteredConfig& config,
               const std::shared_ptr<Metrics>& metrics = nullptr,
               const ov::SoPtr<IEngineBackend>& backend = {nullptr});

    /**
     * @brief Initialize the properties map and try registering the properties for npu-plugin and compiled-model
     * Can be used for both plugin and compiled-model properties maps, based on the provided pType param to the
     * constructor of this object
     * @details
     * - it will reset the properties map
     * - it will try registering config-backed option-based properties, with data from global configuration (supported,
     * visibilty, mutability, value)
     * - if an option is not present in the global config, it assumes it is not supported and will skip it
     * - it will register metric-based properties, with data from the metrics interface
     * - at the end it populates supported_properties with the now dynamically registered public properties
     */
    void registerProperties();

    /**
     * @brief Get the values of a property in a map
     */
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments = {}) const;

    /**
     * @brief Set the values of a subset of properties, provided as a map
     * @details
     * - checks if the property exists, will report if unsupported
     * - checks if the property is Read-only, will report error if so
     */
    void set_property(const ov::AnyMap& properties);

private:
    PropertiesType _pType;
    FilteredConfig& _config;
    std::shared_ptr<Metrics> _metrics;
    ov::SoPtr<IEngineBackend> _backend;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;

    // internal registration functions basd on client object
    void registerPluginProperties();
    void registerCompiledModelProperties();

    const std::vector<ov::PropertyName> _cachingProperties = {
        ov::device::architecture.name(),
        ov::intel_npu::compilation_mode_params.name(),
        ov::intel_npu::compiler_dynamic_quantization.name(),
        ov::intel_npu::tiles.name(),
        ov::intel_npu::dma_engines.name(),
        ov::intel_npu::compilation_mode.name(),
        ov::intel_npu::driver_version.name(),
        ov::intel_npu::compiler_type.name(),
        ov::intel_npu::batch_mode.name(),
        ov::hint::execution_mode.name(),
        ov::cache_mode.name(),
        // NPUW caching properties
        ov::intel_npu::use_npuw.name(),
        ov::intel_npu::npuw::devices.name(),
        ov::intel_npu::npuw::submodel_device.name(),
        ov::intel_npu::npuw::weights_bank.name(),
        ov::intel_npu::npuw::weights_bank_alloc.name(),
        ov::intel_npu::npuw::partitioning::online::pipeline.name(),
        ov::intel_npu::npuw::partitioning::online::avoid.name(),
        ov::intel_npu::npuw::partitioning::online::isolate.name(),
        ov::intel_npu::npuw::partitioning::online::nofold.name(),
        ov::intel_npu::npuw::partitioning::online::min_size.name(),
        ov::intel_npu::npuw::partitioning::online::keep_blocks.name(),
        ov::intel_npu::npuw::partitioning::online::keep_block_size.name(),
        ov::intel_npu::npuw::partitioning::fold.name(),
        ov::intel_npu::npuw::partitioning::cwai.name(),
        ov::intel_npu::npuw::partitioning::dyn_quant.name(),
        ov::intel_npu::npuw::partitioning::dyn_quant_full.name(),
        ov::intel_npu::npuw::partitioning::par_matmul_merge_dims.name(),
        ov::intel_npu::npuw::partitioning::slice_out.name(),
        ov::intel_npu::npuw::partitioning::spatial.name(),
        ov::intel_npu::npuw::partitioning::spatial_nway.name(),
        ov::intel_npu::npuw::partitioning::spatial_dyn.name(),
        ov::intel_npu::npuw::partitioning::f16_interconnect.name(),
        ov::intel_npu::npuw::partitioning::host_gather.name(),
        ov::intel_npu::npuw::partitioning::dcoff_type.name(),
        ov::intel_npu::npuw::partitioning::dcoff_with_scale.name(),
        ov::intel_npu::npuw::partitioning::funcall_for_all.name(),
        ov::intel_npu::npuw::funcall_async.name(),
        ov::intel_npu::npuw::unfold_ireqs.name(),
        ov::intel_npu::npuw::llm::enabled.name(),
        ov::intel_npu::npuw::llm::batch_dim.name(),
        ov::intel_npu::npuw::llm::seq_len_dim.name(),
        ov::intel_npu::npuw::llm::max_prompt_len.name(),
        ov::intel_npu::npuw::llm::max_generation_token_len.name(),
        ov::intel_npu::npuw::llm::min_response_len.name(),
        ov::intel_npu::npuw::llm::optimize_v_tensors.name(),
        ov::intel_npu::npuw::llm::cache_rope.name(),
        ov::intel_npu::npuw::llm::prefill_chunk_size.name(),
        ov::intel_npu::npuw::llm::shared_lm_head.name(),
        ov::intel_npu::npuw::llm::max_lora_rank.name(),
        ov::intel_npu::npuw::llm::prefill_hint.name(),
        ov::intel_npu::npuw::llm::prefill_config.name(),
        ov::intel_npu::npuw::llm::additional_prefill_config.name(),
        ov::intel_npu::npuw::llm::prefill_attn_hint.name(),
        ov::intel_npu::npuw::llm::generate_hint.name(),
        ov::intel_npu::npuw::llm::generate_config.name(),
        ov::intel_npu::npuw::llm::additional_generate_config.name(),
        ov::intel_npu::npuw::llm::generate_attn_hint.name(),
        ov::intel_npu::npuw::llm::shared_lm_head_config.name(),
        ov::intel_npu::npuw::llm::additional_shared_lm_head_config.name()};

    const std::vector<ov::PropertyName> _internalSupportedProperties = {ov::internal::caching_properties.name(),
                                                                        ov::internal::caching_with_mmap.name()};
};

}  // namespace intel_npu
