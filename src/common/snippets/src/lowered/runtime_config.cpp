// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/runtime_config.hpp"

#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

bool RuntimeConfig::contains(size_t loop_id, const LoopDescriptor::Type& type) const {
    OPENVINO_ASSERT(m_loops.count(loop_id) > 0, "LoopId has not been found!");
    auto& loop_descriptors = m_loops.at(loop_id);
    return std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                        [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; }) != loop_descriptors.cend();
}

bool RuntimeConfig::get_loop_desc_it(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptorList::iterator& desc_it) {
    OPENVINO_ASSERT(m_loops.count(loop_id) > 0, "LoopId has not been found!");
    auto& loop_descriptors = m_loops.at(loop_id);
    desc_it = std::find_if(loop_descriptors.begin(), loop_descriptors.end(),
                        [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
    return desc_it != loop_descriptors.end();
}

bool RuntimeConfig::get_last_executed_loop_desc_it(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptorList::iterator& desc_it) {
    OPENVINO_ASSERT(m_loops.count(loop_id) > 0, "LoopId has not been found!");
    auto& loop_descriptors = m_loops.at(loop_id);
    desc_it = loop_descriptors.end();
    for (auto it = loop_descriptors.begin(); it != loop_descriptors.end(); ++it) {
        if (it->type == type)
            break;
        if (!utils::is_dynamic_vdim(it->work_amount) && it->work_amount > 0) {
            desc_it = it;
        }
    }
    return desc_it != loop_descriptors.end();
}

bool RuntimeConfig::get_loop_desc(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptor& desc, size_t& index) const {
    OPENVINO_ASSERT(m_loops.count(loop_id) > 0, "LoopID has not been found!");
    index = 0;
    for (const auto& p : m_loops) {
        const auto& id = p.first;
        const auto& loop_descriptors = p.second;
        if (id == loop_id) {
            const auto desc_it = std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                                      [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
            if (desc_it != loop_descriptors.cend()) {
                desc = *desc_it;
                index += std::distance(loop_descriptors.cbegin(), desc_it);
                return true;
            }
            return false;
        }
        index += loop_descriptors.size();
    }
    return false;
}

bool RuntimeConfig::get_loop_desc(size_t loop_id, const LoopDescriptor::Type& type, LoopDescriptor& desc) const {
    OPENVINO_ASSERT(m_loops.count(loop_id) > 0, "LoopId has not been found!");
    const auto& loop_descriptors = m_loops.at(loop_id);
    const auto desc_it = std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                                      [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
    if (desc_it != loop_descriptors.cend()) {
        desc = *desc_it;
        return true;
    }
    return false;
}

RuntimeConfig::LoopDescriptorList::iterator RuntimeConfig::push_new_desc(size_t loop_id, const LoopDescriptor::Type& type) {
    OPENVINO_ASSERT(m_loops.count(loop_id) > 0, "LoopId has not been found!");
    return m_loops.at(loop_id).insert(m_loops.at(loop_id).cend(), RuntimeConfig::LoopDescriptor(type));
}

size_t RuntimeConfig::get_full_loop_descriptor_count() const {
    return std::accumulate(m_loops.cbegin(), m_loops.cend(), size_t(0),
                           [](size_t count, const std::pair<size_t, LoopDescriptorList>& p) { return count + p.second.size(); });
}

void RuntimeConfig::clear() {
    m_loops.clear();
    m_data_offsets.clear();
}

} // namespace lowered
} // namespace snippets
} // namespace ov
