// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov {
namespace intel_cpu {

class CPURuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    OPENVINO_RTTI("CPURuntimeConfig", "0", ov::snippets::RuntimeConfig)
    CPURuntimeConfig() = default;

    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator();

protected:
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     */
    void update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) override;
    /**
     * @brief Allocate and intialize fields in RuntimeConfig and RuntimeConfigurator
     * @param linear_ir LinearIR
     */
    void initialization(const ov::snippets::lowered::LinearIRPtr& linear_ir) override;
    /**
     * @brief Initializes tensor rank of config
     * @param linear_ir LinearIR
     */
    void init_tensor_rank(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const override;
    /**
     * @brief Calculate Loop parameters of Loop emitters and update these values in CPURuntimeConfig
     * @param linear_ir LinearIR
     */
    void update_loop_args(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const;

    static const size_t rank6D;

    class ParallelWAOptimizer {
    public:
        /**
         * @brief Inits ParallelWAOptimizer: computes optimizer parameters which should be set at compilation stage
         * @param linear_ir LinearIR
         */
        void init(const ov::snippets::lowered::LinearIRPtr& linear_ir);
        /**
         * @brief Defines if the optimizer should be applied
         * @attention It also computes "batch_m" and "new_m" runtime parameters
         * @param master_shape Master shape
         */
        bool need_optimize(const ov::snippets::VectorDims& master_shape);
        /**
         * @brief Updates loops_to_split loop info: sets "new_m" work amount, and correspondingly updates runtime params
         * @param map Loop info -> Runtime params map which will be passed in "update_loop_info"
         */
        void update_split_loops_info(ov::snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap& map);
        /**
         * @brief Splits m dimension in shapes
         * @param io_descs Descriptors which contain original shapes
         * @param shapes Vector which is filled with the split shapes
         * @param in_num Number of inputs
         */
        void update_shapes(const std::vector<snippets::lowered::PortDescriptorPtr>& io_descs,
                           std::vector<ov::snippets::VectorDims>& shapes,
                           size_t in_num);
        /**
         * @brief Splits m dimension in layouts
         * @param io_descs Descriptors which contain original layouts
         * @param shapes Vector which is filled with the split layouts
         * @param in_num Number of inputs
         */
        void update_layouts(const std::vector<snippets::lowered::PortDescriptorPtr>& io_descs,
                            std::vector<std::vector<size_t>>& layouts,
                            size_t in_num);
        /**
         * @brief Updates runtime config: tile rank and master shape
         * @param config Config which should be updated
         */
        void update_config(const std::shared_ptr<ov::snippets::RuntimeConfig>& config);

    private:
        static std::unordered_set<snippets::lowered::ExpressionPtr> find_applicable_brgemms(const ov::snippets::lowered::LinearIRPtr& linear_ir);
        void init_non_m_related_params(const ov::snippets::lowered::LinearIRPtr& linear_ir,
                                       const std::unordered_set<snippets::lowered::ExpressionPtr>& brgemms);
        void init_loops_to_split(const ov::snippets::lowered::LinearIRPtr& linear_ir);

        std::unordered_set<ov::snippets::lowered::UnifiedLoopInfoPtr> loops_to_split{};
        std::unordered_set<size_t> not_m_related_params{};
        size_t concurrency = 0;
        size_t batch_m = 0;
        size_t new_m = 0;
    } m_optimizer;
};

}   // namespace intel_cpu
}   // namespace ov
