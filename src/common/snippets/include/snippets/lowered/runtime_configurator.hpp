// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/runtime_config.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"


namespace ov {
namespace snippets {
namespace lowered {

/**
 * @interface RuntimeConfigurator
 * @brief Describes the runtime-dependent (shape-dependent) information using Runtime Config
 */
class RuntimeConfigurator {
public:
    RuntimeConfigurator() = default;

    /**
     * @brief Initialize config using LinearIR state
     * @param linear_ir the updated LinearIR
     */
    void update(const lowered::LinearIR& linear_ir);
    /**
     * @brief Reset config: remove all loop descriptors
     */
    void reset();

    /**
     * @brief Get runtime config
     * @return config with runtime parameters
     */
    RuntimeConfig get_config() const { return m_config; }

private:
    using LinearIR = lowered::LinearIR;
    /**
     * @brief Initialize the map of loops descriptors using LoopManager of LinearIR
     * @param loop_manager LoopManager of needed LinearIR
     */
    void init_loop_descriptors(const LinearIR::LoopManagerPtr& loop_manager);
    /**
     * @brief Initialize input and output data offsets
     * @param linear_ir Linear IR
     */
    void init_data_offsets(const LinearIR& linear_ir);
    /**
     * @brief Calculate offset for input or output of body
     * @param desc port descriptor
     * @param data_size byte size of data type of input or output
     * @param is_input true if it's input otherwise it's false
     * @param rank common tensor rank
     * @param offsets reference on the target offsets for update
     */
    static void offset_calculation(const lowered::PortDescriptorPtr& desc, size_t data_size, bool is_input, size_t rank, std::vector<size_t>& offsets);
    /**
     * @brief Initialize the first iteration loop descriptor
     * @param loop_info loop information of the corresponding loop
     * @param loop_id id of the target Loop
     * @param first_iter_loop_desc ref of the first iter loop descriptor which should be inited
     */
    void init_first_iter_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                         RuntimeConfig::LoopDescriptor& first_iter_loop_desc);
    /**
     * @brief Initialize the vector loop descriptor
     * @param loop_info loop information of the corresponding loop
     * @param loop_id id of the target Loop
     * @param vector_loop_desc ref of the vector loop descriptor which should be inited
     */
    void init_vector_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                     RuntimeConfig::LoopDescriptor& vector_loop_desc);
    /**
     * @brief Initialize the tail loop descriptor
     * @param loop_info loop information of the corresponding loop
     * @param loop_id id of the target Loop
     * @param tail_loop_desc ref of the tail loop descriptor which should be inited
     */
    void init_tail_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                   RuntimeConfig::LoopDescriptor& tail_loop_desc);
    /**
     * @brief Initialize the inner tail splited loops
     * @param loop_manager LoopManager of needed LinearIR
     * @param outer_splited_loop_info loop information of the outer splited loop
     * @param outer_splited_tail_loop_desc tail descriptor of the outer splited loop
     * @param outer_loop_id ID of the outer splited loop
     */
    void init_inner_splited_tail_loop_descriptors(const LinearIR::LoopManagerPtr& loop_manager,
                                                  const LinearIR::LoopManager::LoopInfoPtr& outer_splited_loop_info,
                                                  const RuntimeConfig::LoopDescriptor& outer_splited_tail_loop_desc,
                                                  size_t outer_loop_id);
    /**
     * @brief Initialize ptr increments and finalization offsets from LoopPorts.
     *        If there is previous Loop iteration (another Loop body before),
     *        moves shifts frm the previous to the current and zeros them in the previous body.
     * @param desc target Loop Descriptor
     * @param loop_ports ports of the target Loop taht contains ptr increments and finalization offsets
     * @param skip_evaluation true if work amount is zero and data shifts should be inited by zero
     * @param is_there_prev_iter true if there is executed loop iterations before
     * @param prev_iter_desc iterator to the descriptor of the previos loop body
     */
    void init_data_ptr_shifts(RuntimeConfig::LoopDescriptor& desc,
                              const std::vector<LinearIR::LoopManager::LoopPort>& loop_ports,
                              bool skip_evaluation, bool is_there_prev_iter,
                              const RuntimeConfig::LoopDescriptorList::iterator& prev_iter_desc);

    /**
     * @brief Check if first iter is needed
     * @param loop_info loop information of the corresponding loop
     * @return True if needed otherwise returns False
     */
    inline static bool is_first_iter_loop_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
        return loop_info->get_first_iter_handler() != nullptr && (loop_info->get_work_amount() >= loop_info->get_increment() || loop_info->is_dynamic());
    }
    /**
     * @brief Check if vector loop is needed
     * @param loop_info loop information of the corresponding loop
     * @return True if needed otherwise returns False
     */
    inline static bool is_vector_loop_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
        return (is_first_iter_loop_needed(loop_info) ? loop_info->get_work_amount() >= 2 * loop_info->get_increment()
                                                     : loop_info->get_work_amount() >= loop_info->get_increment()) ||
                loop_info->is_dynamic();
    }
    /**
     * @brief Check if tail loop is needed
     * @param loop_info loop information of the corresponding loop
     * @return True if needed otherwise returns False
     */
    inline static bool is_tail_loop_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
        return (loop_info->get_work_amount() % loop_info->get_increment() != 0) || (loop_info->is_dynamic() && loop_info->get_increment() > 1);
    }

    RuntimeConfig m_config;
    bool m_is_first_init = true;
};

} // namespace lowered
} // namespace snippets
} // namespace ov