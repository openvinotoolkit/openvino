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
    class LoopInitializer;
    class FirstLoopInitializer;
    class MainLoopInitializer;
    class LastLoopInitializer;

public:
    RuntimeConfigurator();

    /**
     * @brief Initialize config using LinearIR state
     * @param linear_ir the updated LinearIR
     */
    void init(const lowered::LinearIR& linear_ir);
    /**
     * @brief Initialize config using LinearIR state
     * @param linear_ir the updated LinearIR
     */
    void update(const lowered::LinearIR& linear_ir);
    /**
     * @brief Clone with new LinearIR info with loop descs, data offsets savings
     * @param linear_ir the updated LinearIR
     */
    std::shared_ptr<RuntimeConfigurator> clone(const lowered::LinearIR& linear_ir);
    /**
     * @brief Reset config: remove all loop descriptors
     */
    void reset();

    /**
     * @brief Get runtime config
     * @return config with runtime parameters
     */
    const RuntimeConfig& get_config() const { return m_config; }

private:
    using LinearIR = lowered::LinearIR;
    /**
     * @brief Initialize the map of loops descriptors using LoopManager of LinearIR
     * @param loop_manager LoopManager of needed LinearIR
     */
    void init_loop_descriptors(const LinearIR::LoopManagerPtr& loop_manager);
    /**
     * @brief Initialize input and output data offsets
     */
    void init_data_offsets();
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
     * @brief Initialize input and output of LinearIR descriptors and data sizes.
     * @param linear_ir target linear ir
     */
    void init_io_info(const LinearIR& linear_ir);

    size_t m_io_num = 0;
    size_t m_in_num = 0;
    size_t m_tensor_rank = 0;
    std::vector<PortDescriptorPtr> m_io_descs = {};
    std::vector<size_t> m_io_data_sizes = {};

    std::map<RuntimeConfig::LoopDescriptor::Type, std::shared_ptr<RuntimeConfigurator::LoopInitializer>> m_desc_initializers;
    RuntimeConfig m_config;
    bool m_inited = false;
};


class RuntimeConfigurator::LoopInitializer {
public:
    LoopInitializer(bool& inited) : m_inited(inited) {}

    /**
     * @brief Check if specific loop is needed
     * @param loop_info loop information of the corresponding loop
     * @return True if needed otherwise returns False
     */
    virtual bool is_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) = 0;

    virtual void init_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                                 const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                 RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) = 0;

protected:
    /**
     * @brief Initialize ptr increments and finalization offsets from LoopPorts.
     *        If there is previous Loop iteration (another Loop body before),
     *        moves shifts frm the previous to the current and zeros them in the previous body.
     * @param loop_ports ports of the target Loop taht contains ptr increments and finalization offsets
     * @param skip_evaluation true if work amount is zero and data shifts should be inited by zero
     * @param is_there_prev_iter true if there is executed loop iterations before
     * @param prev_iter_desc iterator to the descriptor of the previos loop body
     * @param desc target Loop Descriptor
     */
    void init_data_ptr_shifts(const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                              bool skip_evaluation, bool is_there_prev_iter,
                              const RuntimeConfig::LoopDescriptorList::iterator& prev_iter_desc,
                              RuntimeConfig::LoopDescriptor& desc);

    bool& m_inited;
};

class RuntimeConfigurator::FirstLoopInitializer : public RuntimeConfigurator::LoopInitializer {
public:
    FirstLoopInitializer(bool& inited) : LoopInitializer(inited) {}

    bool is_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) override;

    void init_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                         const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                         RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) override;
};

class RuntimeConfigurator::MainLoopInitializer : public RuntimeConfigurator::LoopInitializer {
public:
    MainLoopInitializer(bool& inited) : LoopInitializer(inited) {}

    bool is_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) override;

    void init_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                         const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                         RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) override;
};

class RuntimeConfigurator::LastLoopInitializer : public RuntimeConfigurator::LoopInitializer {
public:
    LastLoopInitializer(bool& inited) : LoopInitializer(inited) {}

    bool is_needed(const LinearIR::LoopManager::LoopInfoPtr& loop_info) override;

    void init_descriptor(const LinearIR::LoopManagerPtr& loop_manager,
                         const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                         RuntimeConfig::LoopDescriptor& desc, RuntimeConfig& config) override;

private:
    /**
     * @brief Initialize the inner tail splited loops
     * @param loop_manager LoopManager of needed LinearIR
     * @param outer_splited_loop_info loop information of the outer splited loop
     * @param outer_splited_tail_loop_desc tail descriptor of the outer splited loop
     * @param outer_loop_id ID of the outer splited loop
     */
    void init_inner_splited_descriptors(const LinearIR::LoopManagerPtr& loop_manager,
                                        const LinearIR::LoopManager::LoopInfoPtr& outer_splited_loop_info,
                                        const RuntimeConfig::LoopDescriptor& outer_splited_tail_loop_desc,
                                        size_t outer_loop_id, RuntimeConfig& config);
};

} // namespace lowered
} // namespace snippets
} // namespace ov