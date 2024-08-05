// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/kernel_executor_table.hpp"
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
    std::vector<size_t> buffer_cluster_offsets {};
    KernelExecutorTablePtr kernel_executor_table = std::make_shared<ov::snippets::KernelExecutorTable>();
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
    const std::shared_ptr<RuntimeConfig>& get_updated_config(const lowered::LinearIRCPtr& linear_ir);
    /**
     * @brief Returns pointer to KernelExecutorTable owned by the config
     * @return updated KernelExecutorTable
     */
    const std::shared_ptr<KernelExecutorTable>& get_kernel_executor_table() const { return m_config->kernel_executor_table; }
    /**
     * @brief Set new KernelExecutorTable to the config
     * @param table new KernelExecutorTable
     */
    void set_kernel_executor_table(std::shared_ptr<KernelExecutorTable> table) const;

    /**
     * @brief Reset KernelExecutor table
     */
    void reset_kernel_executor_table() const;

protected:
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     */
    virtual void update(const lowered::LinearIRCPtr& linear_ir);
    /**
     * @brief Allocate and intialize fields in RuntimeConfig and RuntimeConfigurator
     * @param linear_ir LinearIR
     */
    virtual void initialization(const lowered::LinearIRCPtr& linear_ir);

    /**
     * @brief Initializes input and data information of LinearIR:
     *        descriptors (that contains shapes and layouts) and data_sizes
     * @param linear_ir LinearIR
     */
    void init_data_info(const lowered::LinearIRCPtr& linear_ir);
    /**
     * @brief Initializes information of buffers:
     *        - static buffer_scratchpad_size
     *        - offsets of static clusters (with static buffers)
     *        - clusters with dynamic buffers (`m_dynamic_buffer_clusters`) for the quick access in `update()`
     * @param linear_ir LinearIR
     */
    void init_buffer_info(const lowered::LinearIRCPtr& linear_ir);
    /**
     * @brief Initializes tensor rank of config
     * @param linear_ir LinearIR
     */
    virtual void init_tensor_rank(const lowered::LinearIRCPtr& linear_ir) const;

    struct UnifiedLoopInfoRtParams {
        size_t work_amount = 0;
        std::vector<int64_t> ptr_increments;
        std::vector<int64_t> finalization_offsets;
    };
    static UnifiedLoopInfoRtParams compute_runtime_params(const lowered::UnifiedLoopInfoPtr& unified_loop_info);
    using LoopInfoRuntimeParamsMap = std::unordered_map<lowered::UnifiedLoopInfoPtr, UnifiedLoopInfoRtParams>;
    /**
     * @brief Update Loop informations in LinearIR: Unified and ExpandedLoopInfo
     * @param linear_ir LinearIR
     * @param initializated_info_map Reference on a map [LoopInfo->RuntimeParams].
     * Can be used to pass in the method loop infos which were already initialized, e.g. by parallel domain optimization
     */
    void update_loop_info(const lowered::LinearIRCPtr& linear_ir, LoopInfoRuntimeParamsMap& initializated_info_map) const;
    /**
     * @brief Update Buffer scratchpad size and offsets if needed
     *        Note: `update_loop_info` must be called before
     * @param linear_ir LinearIR
     */
    void update_buffer_scratchpad_size(const lowered::LinearIRCPtr& linear_ir) const;
    /**
     * @brief Calculate data offsets of LinearIR and update these values in RuntimeConfig
     * @param shapes shapes used in offsets computation
     * @param layouts layouts used in offsets computation
     */
    void update_data_offsets(const std::vector<ov::snippets::VectorDims>& shapes,
                             const std::vector<std::vector<size_t>>& layouts) const;
    /**
     * @brief Extract shapes from m_io_descs
     */
    std::vector<ov::snippets::VectorDims> extract_shapes();
    /**
     * @brief Extract layouts from m_io_descs
     */
    std::vector<std::vector<size_t>> extract_layouts();

    class ParallelWAOptimizer {
    public:
        /**
         * @brief Inits ParallelWAOptimizer: computes optimizer parameters which should be set at compilation stage
         * @param linear_ir LinearIR
         */
        void init(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                  const std::vector<snippets::lowered::PortDescriptorPtr>& io_descs,
                  size_t in_num);
        /**
         * @brief Defines if the optimizer should be applied
         * @attention It also computes "batch_m" and "new_m" runtime parameters
         * @param master_shape Master shape
         */
        bool need_optimize(const ov::snippets::VectorDims& master_shape);
        /**
         * @brief Updates all the necessary runtime information
         * @param map Loop info -> Runtime params map which will be passed in "update_loop_info"
         * the map is filled with updated loops_to_split loops: "new_m" work amount is set for them, and runtime params are updated correspondingly
         * @param shapes Vector which is filled with the split shapes
         * @param layouts Vector which is filled with the split layouts
         * @param config Config which should be updated: tile rank and master shape is updated
         * @attention It also computes "batch_m" and "new_m" runtime parameters
         * @param master_shape Master shape
         */
        void update(ov::snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap& map,
                    std::vector<ov::snippets::VectorDims>& shapes,
                    std::vector<std::vector<size_t>>& layouts,
                    const std::shared_ptr<ov::snippets::RuntimeConfig>& config,
                    size_t in_num);

    private:
        void update_split_loops_info(ov::snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap& map);
        void update_shapes(std::vector<ov::snippets::VectorDims>& shapes);
        void update_layouts(std::vector<std::vector<size_t>>& layouts);
        void update_config(const std::shared_ptr<ov::snippets::RuntimeConfig>& config);

        static std::unordered_set<snippets::lowered::ExpressionPtr> find_applicable_brgemms(const ov::snippets::lowered::LinearIRCPtr& linear_ir);
        void init_non_m_related_params(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                       const std::unordered_set<snippets::lowered::ExpressionPtr>& brgemms);
        void init_loops_to_split(const ov::snippets::lowered::LinearIRCPtr& linear_ir);

        std::unordered_set<ov::snippets::lowered::UnifiedLoopInfoPtr> loops_to_split{};
        std::unordered_set<size_t> not_m_related_params{};
        std::vector<std::vector<size_t>> split_layouts{};
        std::vector<size_t> m_dim_idces{};
        size_t concurrency = 0;
        size_t batch_m = 0;
        size_t new_m = 0;
    } m_optimizer;

    std::shared_ptr<RuntimeConfig> m_config = nullptr;

    size_t m_io_num = 0;
    size_t m_in_num = 0;
    std::vector<snippets::lowered::PortDescriptorPtr> m_io_descs = {};
    std::vector<size_t> m_io_data_sizes = {};
    // [cluster_id -> buffer expressions ]
    std::map<size_t, std::set<lowered::ExpressionPtr>> m_dynamic_buffer_clusters;

    std::vector<ov::snippets::VectorDims> m_latest_shapes = {};
};

} // namespace snippets
} // namespace ov
