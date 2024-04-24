// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/runtime_configurator.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"

namespace ov {
namespace intel_cpu {

class CPURuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    CPURuntimeConfig() = default;

    size_t tensor_rank = 0;
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
    std::vector<ov::snippets::VectorDims> io_data_offsets = {};
    ov::snippets::VectorDims parallel_domain = {};
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator();

protected:
    /**
     * @brief Return `True` if input shapes of LinearIR have been updated and config should be updated.
     *        Otherwise returns `False`
     * @param linear_ir LinearIR
     * @return boolean
     */
    bool is_update_needed(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) override;
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     */
    void update(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) override;

    /**
     * @brief Initializes input and data information of LinearIR: layouts, shapes, data_sizes
     * @param linear_ir LinearIR
     */
    void init_data_info(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir);
    /**
     * @brief Calculate data offsets of LinearIR and update these values in CPURuntimeConfig
     * @param cpu_config CPURuntimeConfig
     */
    void update_data_offsets(const std::shared_ptr<CPURuntimeConfig>& cpu_config) const;
    /**
     * @brief Calculate Loop parameters of Loop emitters and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     * @param cpu_config CPURuntimeConfig
     */
    void update_loop_args(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                          const std::shared_ptr<CPURuntimeConfig>& cpu_config) const;
    /**
     * @brief Calculate parallel domain and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     * @param cpu_config CPURuntimeConfig
     */
    void update_parallel_domain(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                                const std::shared_ptr<CPURuntimeConfig>& cpu_config) const;
    /**
     * @brief Update latest input shapes
     */
    void update_latest_shapes();

    const size_t rank6D = 6;

    size_t m_io_num = 0;
    size_t m_in_num = 0;
    std::vector<ov::snippets::VectorDimsPtr> m_io_shapes = {};
    std::vector<std::vector<size_t>> m_io_layouts = {};
    std::vector<size_t> m_io_data_sizes = {};

    std::vector<ov::snippets::VectorDims> m_latest_input_shapes = {};
};

}   // namespace intel_cpu
}   // namespace ov
