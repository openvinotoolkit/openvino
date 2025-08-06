// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"

namespace ov::snippets {

/**
 * @interface RuntimeConfig
 * @brief The config that contains information about LinearIR in runtime.
 */
class RuntimeConfig {
public:
    OPENVINO_RTTI_BASE("RuntimeConfig")

    RuntimeConfig() = default;
    virtual ~RuntimeConfig() = default;

    [[nodiscard]] const char* get_type_name() const {
        return get_type_info().name;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    [[nodiscard]] virtual std::string to_string() const;
#endif

    size_t tensor_rank = 0;
    size_t tile_rank = 0;

    std::vector<ov::snippets::VectorDims> io_shapes;
    std::vector<ov::snippets::VectorDims> io_layouts;
    std::vector<ov::snippets::VectorDims> io_data_offsets;
    ov::snippets::VectorDims master_shape;

    size_t buffer_scratchpad_size = 0;
    std::vector<size_t> buffer_cluster_offsets;
    KernelExecutorTablePtr kernel_executor_table = std::make_shared<ov::snippets::KernelExecutorTable>();
    std::vector<ov::snippets::VectorDims> latest_shapes;
};

/**
 * @interface RuntimeConfigurator
 * @brief Configure runtime config based on runtime information of LinearIR
 */
class RuntimeConfigurator {
public:
    explicit RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c);
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
    [[nodiscard]] const std::shared_ptr<KernelExecutorTable>& get_kernel_executor_table() const {
        return m_config->kernel_executor_table;
    }
    /**
     * @brief Set new KernelExecutorTable to the config
     * @param table new KernelExecutorTable
     */
    void set_kernel_executor_table(std::shared_ptr<KernelExecutorTable> table) const;

    /**
     * @brief Reset KernelExecutor table
     */
    void reset_kernel_executor_table() const;

    // Getters for private members
    [[nodiscard]] std::shared_ptr<RuntimeConfig> get_config() const {
        return m_config;
    }
    [[nodiscard]] size_t get_io_num() const {
        return m_io_num;
    }
    [[nodiscard]] size_t get_in_num() const {
        return m_in_num;
    }
    [[nodiscard]] const std::vector<snippets::lowered::PortDescriptorPtr>& get_io_descs() const {
        return m_io_descs;
    }
    [[nodiscard]] const std::vector<size_t>& get_io_data_sizes() const {
        return m_io_data_sizes;
    }
    [[nodiscard]] const std::map<size_t, std::set<lowered::BufferExpressionPtr>>& get_dynamic_buffer_clusters() const {
        return m_dynamic_buffer_clusters;
    }

    /**
     * @brief Computes the offsets for each dimension of a tensor shape.
     *
     * This function calculates the offsets for each dimension of a tensor shape, which represent the distance between
     * consecutive elements of the corresponding dimension. If a dimension size is 1, the next dimension starts
     * immediately, and the stride is 0.
     * @param shape The shape for offset computation.
     * @param idx The index to get the corresponding offsets and io_data_sizes.
     * @param idx_stride Defines the number of dimensions that should be skipped in the offsets vector.
     */
    void compute_offsets(const ov::snippets::VectorDims& shape, size_t idx, size_t idx_stride) const;
    struct UnifiedLoopInfoRtParams {
        size_t work_amount = 0;
        std::vector<int64_t> ptr_increments;
        std::vector<int64_t> finalization_offsets;
    };
    /**
     * @brief Retrieves the runtime parameters for a given UnifiedLoopInfo.
     * @param unified_loop_info The UnifiedLoopInfo for which the runtime parameters are to be retrieved.
     * @return A LoopInfoRuntimeParams object containing the runtime parameters.
     */
    static UnifiedLoopInfoRtParams get_loop_runtime_params(const lowered::UnifiedLoopInfoPtr& unified_loop_info);
    using LoopInfoRuntimeParamsMap = std::unordered_map<lowered::UnifiedLoopInfoPtr, UnifiedLoopInfoRtParams>;
    /**
     * @brief Update Loop information in LinearIR: Unified and ExpandedLoopInfo
     * @param linear_ir LinearIR
     */
    static void update_loop_info(const lowered::LinearIRCPtr& linear_ir);
    /**
     * @brief Updates the ExpandedLoopInfo based on the initialized runtime parameters.
     * @param expanded_loop_info The ExpandedLoopInfo to be updated.
     * @param initialized_info_map A map containing the initialized runtime parameters for UnifiedLoopInfo.
     */
    static void update_expanded_loop_info(const lowered::ExpandedLoopInfoPtr& expanded_loop_info,
                                          LoopInfoRuntimeParamsMap& initialized_info);
    /**
     * @brief Update tensor rank based on master shape
     * @param master_shape Master shape
     */
    virtual void update_tensor_rank(const ov::snippets::VectorDims& master_shape) const;

protected:
    /**
     * @brief Update RuntimeConfig based on LinearIR
     * @param linear_ir LinearIR
     * @todo Ticket 148891: Rewrite on PassPipeline
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
    void update_data_offsets() const;
    /**
     * @brief Extract shapes from m_io_descs
     */
    [[nodiscard]] std::vector<ov::snippets::VectorDims> extract_shapes() const;
    /**
     * @brief Extract layouts from m_io_descs
     */
    [[nodiscard]] std::vector<std::vector<size_t>> extract_layouts() const;

    std::shared_ptr<RuntimeConfig> m_config = nullptr;

    size_t m_io_num = 0;
    size_t m_in_num = 0;
    std::vector<snippets::lowered::PortDescriptorPtr> m_io_descs;
    std::vector<size_t> m_io_data_sizes;
    // [cluster_id -> buffer expressions ]
    std::map<size_t, std::set<lowered::BufferExpressionPtr>> m_dynamic_buffer_clusters;

    // WA: until ticket 148891 is not implemented, 2 pass pipelines for runtime optimizers are necessary since different
    // optimizers must be called at different pipeline stages.
    // - Intermediate optimizers must be called right after `update_loop_info`
    // - Final optimizers must be called after all other RuntimeConfigurator's update methods
    // When all updates will be rewritten on PassPipeline, PositionedPasses can be used to precisely define the place of
    // the additional optimizers
    lowered::pass::PassPipeline m_intermediate_optimizers;
    lowered::pass::PassPipeline m_final_optimizers;
};

}  // namespace ov::snippets
