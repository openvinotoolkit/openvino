// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace snippets {

/**
 * @interface RuntimeConfig
 * @brief The config that contains information about LinearIR in runtime.
 */
class RuntimeConfig {
public:
    RuntimeConfig() = default;
    virtual ~RuntimeConfig() = default;

    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static {"RuntimeConfig"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }

    size_t tensor_rank = 0;
    size_t tile_rank = 0;
    std::vector<ov::snippets::VectorDims> io_data_offsets = {};
    ov::snippets::VectorDims master_shape = {};
    size_t buffer_scratchpad_size = 0;
};

/**
 * @interface RuntimeConfigurator
 * @brief Configure runtime config based on runtime information of LinearIR
 */
class RuntimeConfigurator {
public:
    RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c);
    virtual ~RuntimeConfigurator() = default;

    /**
     * @brief Update RuntimeConfig based on new state of LinearIR and return its
     * @param linear_ir LinearIR
     * @return updated config
     */
    const std::shared_ptr<RuntimeConfig>& get_updated_config(const std::shared_ptr<lowered::LinearIR>& linear_ir);

protected:
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     */
    virtual void update(const std::shared_ptr<lowered::LinearIR>& linear_ir);
    /**
     * @brief Allocate and intialize fields in RuntimeConfig
     * @param linear_ir LinearIR
     */
    virtual void initialization(const std::shared_ptr<lowered::LinearIR>& linear_ir);

    /**
     * @brief Initializes input and data information of LinearIR:
     *        descriptors (that contains shapes and layouts) and data_sizes
     * @param linear_ir LinearIR
     */
    void init_data_info(const std::shared_ptr<lowered::LinearIR>& linear_ir);
    /**
     * @brief Initializes tensor rank of config
     * @param linear_ir LinearIR
     */
    virtual void init_tensor_rank(const std::shared_ptr<lowered::LinearIR>& linear_ir) const;
    /**
     * @brief Calculate data offsets of LinearIR and update these values in RuntimeConfig
     */
    void update_data_offsets() const;
    /**
     * @brief Update latest input shapes
     */
    void update_latest_shapes();

    std::shared_ptr<RuntimeConfig> m_config = nullptr;
    lowered::pass::PassPipeline m_state_updater = {};

    size_t m_io_num = 0;
    size_t m_in_num = 0;
    std::vector<snippets::lowered::PortDescriptorPtr> m_io_descs = {};
    std::vector<size_t> m_io_data_sizes = {};

    std::vector<ov::snippets::VectorDims> m_latest_shapes = {};
};

} // namespace snippets
} // namespace ov
