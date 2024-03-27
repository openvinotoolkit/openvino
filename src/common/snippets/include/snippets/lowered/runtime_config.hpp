// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

class RuntimeConfigurator;

/**
 * @interface RuntimeConfig
 * @brief Contain the runtime-dependent (shape-dependent) information: Loop parameters, offsets of data, buffer size
 */
class RuntimeConfig {
    friend class RuntimeConfigurator;
public:
    RuntimeConfig() = default;

    /**
     * @interface LoopDescriptor
     * @brief Describes Loops - the simple copy of LoopManager::LoopInfo without loop ports
     */
    struct LoopDescriptor {
        enum class Type { First, Main, Last };
        LoopDescriptor() = default;
        LoopDescriptor(Type type, size_t id) : type(type), id(id) {}
        LoopDescriptor(size_t wa, size_t inc, std::vector<int64_t> ptr_incs = {}, std::vector<int64_t> final_offs = {}, std::vector<int64_t> ds = {},
                       Type type = Type::Main, size_t id = 0)
            : work_amount(wa), increment(inc), ptr_increments(std::move(ptr_incs)), finalization_offsets(std::move(final_offs)),
              data_sizes(std::move(ds)), type(type), id(id) {}

        size_t work_amount = utils::get_dynamic_value<size_t>();
        size_t increment = 1;
        std::vector<int64_t> ptr_increments = {};
        std::vector<int64_t> finalization_offsets = {};
        std::vector<int64_t> data_sizes = {};
        Type type = Type::Main;
        size_t id = 0;
    };
    using LoopDescriptorList = std::vector<LoopDescriptor>;
    // [loop_id -> loop descriptors]
    using LoopMap = std::map<size_t, LoopDescriptorList>;

    /**
     * @brief Check if config contains the LoopDescriptors by Loop ID.
     * @param loop_id the corresponding loop ID
     * @return True if there are loop descriptors. Otherwise returns False
     */
    bool contains(size_t loop_id) const;
    /**
     * @brief Check if config contains the LoopDescriptor by Loop ID and Type of descriptor.
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @return True if the loop descriptor has been found. Otherwise returns False
     */
    bool contains(size_t loop_id, const LoopDescriptor::Type& type) const;
    /**
     * @brief Find the LoopDescriptor by Loop ID and Type of descriptor.
     *        Since the method doesn't return ordered index of LoopDescriptor, this method is faster than previous.
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @param desc the reference of the target loop descriptor
     * @return True if the loop descriptor has been found. Otherwise returns False
     */
    bool get_loop_desc(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptor& desc) const;
    /**
     * @brief Return the loop descriptors
     * @return the const ref of the map [loop_id -> loop descriptors]
     */
    const LoopMap& get_loops() const { return m_loops; }
    /**
     * @brief Return the Subgraph input and output data offsets
     * @return the const ref of vector with data offsets
     */
    const std::vector<std::vector<size_t>>& get_data_offsets() const { return m_data_offsets; }
    /**
     * @brief Return the count of all loop descriptors
     * @return the count
     */
    size_t get_loop_descriptor_count() const { return m_loop_desc_count; }
    /**
     * @brief Remove all the existing data from config
     */
    void clear();

private:
    /**
     * @brief Find the last executed LoopDescriptor (work_amount != 0) iterator by Loop ID before the target Loop with Type of descriptor.
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop before which we find the executed loop descriptor
     * @param desc_it Iterator of vector with LoopDescriptors
     * @return status - True if the iterator has been found, otherwise - False
     */
    bool get_last_executed_loop_desc_it(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptorList::iterator& desc_it);
    /**
     * @brief Add to the end new empty descriptor of `type`
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @return Iterator of vector with LoopDescriptors
     */
    LoopDescriptorList::iterator push_new_desc(size_t loop_id, const LoopDescriptor::Type& type);

    // [loop_id -> loop descriptors]
    LoopMap m_loops = {};
    size_t m_loop_desc_count = 0;
    // offsets of subgraph input and output data
    std::vector<std::vector<size_t>> m_data_offsets = {};
};

} // namespace lowered
} // namespace snippets
} // namespace ov
