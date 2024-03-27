// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/runtime_config.hpp"

#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

bool RuntimeConfig::contains(size_t loop_id) const {
    return m_loops.count(loop_id) > 0;
}

bool RuntimeConfig::contains(size_t loop_id, const LoopDescriptor::Type& type) const {
    OPENVINO_ASSERT(contains(loop_id), "LoopId has not been found!");
    auto& loop_descriptors = m_loops.at(loop_id);
    return std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                        [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; }) != loop_descriptors.cend();
}

bool RuntimeConfig::get_loop_desc(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptor& desc) const {
    OPENVINO_ASSERT(contains(loop_id), "LoopId has not been found!");
    auto& loop_descriptors = m_loops.at(loop_id);
    const auto desc_it = std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                                     [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
    if (desc_it != loop_descriptors.cend()) {
        desc = *desc_it;
        return true;
    }
    return false;
}

bool RuntimeConfig::get_last_executed_loop_desc_it(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptorList::iterator& desc_it) {
    OPENVINO_ASSERT(contains(loop_id), "LoopId has not been found!");
    auto& loop_descriptors = m_loops.at(loop_id);
    desc_it = loop_descriptors.end();
    for (auto it = loop_descriptors.begin(); it != loop_descriptors.end(); ++it) {
        if (it->type == type)
            break;
        if (!utils::is_dynamic_value(it->work_amount) && it->work_amount > 0) {
            desc_it = it;
        }
    }
    return desc_it != loop_descriptors.end();
}

RuntimeConfig::LoopDescriptorList::iterator RuntimeConfig::push_new_desc(size_t loop_id, const LoopDescriptor::Type& type) {
    OPENVINO_ASSERT(contains(loop_id), "LoopId has not been found!");
    return m_loops[loop_id].insert(m_loops.at(loop_id).cend(), RuntimeConfig::LoopDescriptor(type, m_loop_desc_count++));
}

void RuntimeConfig::clear() {
    m_loops.clear();
    m_data_offsets.clear();
    m_loop_desc_count = 0;
}

} // namespace lowered
} // namespace snippets
} // namespace ov
