// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/shape_inference/shape_inference.hpp"


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
    /**
     * @interface LoopDescriptor
     * @brief Describes Loops - the simple copy of LoopManager::LoopInfo without loop ports
     */
    struct LoopDescriptor {
        enum Type { First, Vector, Tail, SplitedTail };
        LoopDescriptor() = default;
        LoopDescriptor(size_t wa, size_t inc, std::vector<int64_t> ptr_incs = {}, std::vector<int64_t> final_offs = {}, Type type = Type::Vector)
            : work_amount(wa), increment(inc), ptr_increments(ptr_incs), finalization_offsets(final_offs), type(type) {}

        size_t work_amount = IShapeInferSnippets::DYNAMIC_DIMENSION;
        size_t increment = 1;
        std::vector<int64_t> ptr_increments = {};  // in bytes
        std::vector<int64_t> finalization_offsets = {};  // in bytes
        Type type = Type::Vector;
    };
    using LoopDescriptorList = std::vector<LoopDescriptor>;
    // [loop_id -> loop descriptors]
    using LoopMap = std::map<size_t, LoopDescriptorList>;

    /**
     * @brief Check if config contains the LoopDescriptor by Loop ID and Type of descriptor.
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @return True if the loop descriptor has been found. Otherwise returns False
     */
    bool contains(size_t loop_id, const LoopDescriptor::Type& type) const;
    /**
     * @brief Find the LoopDescriptor and its ordered index by Loop ID and Type of descriptor
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @param desc the reference of the target loop descriptor
     * @param index the reference of ordered index of loop in all loop descriptors
     * @return True if the loop descriptor has been found. Otherwise returns False
     */
    bool get_loop_desc(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptor& desc, size_t& index) const;
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
     * @brief Find the LoopDescriptor iterator by Loop ID and Type of descriptor.
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @return Iterator of vector with LoopDescriptors
     */
    LoopDescriptorList::iterator get_loop_desc_it(size_t loop_id, const LoopDescriptor::Type& type);

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
    size_t get_full_loop_descriptor_count() const;
    /**
     * @brief Remove all the existing data from config
     */
    void clear();

private:
    RuntimeConfig() = default;

    /**
     * @brief Add to the end new empty descriptor of `type`
     * @param loop_id the corresponding loop ID
     * @param type the type of Loop
     * @return Iterator of vector with LoopDescriptors
     */
    LoopDescriptorList::iterator push_new_desc(size_t loop_id, const LoopDescriptor::Type& type);

    // [loop_id -> loop descriptors]
    LoopMap m_loops;
    // offsets of subgraph input and output data
    std::vector<std::vector<size_t>> m_data_offsets;
};

} // namespace lowered
} // namespace snippets
} // namespace ov