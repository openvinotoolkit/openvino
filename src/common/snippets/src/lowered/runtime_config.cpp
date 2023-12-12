// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/runtime_config.hpp"


namespace ov {
namespace snippets {
namespace lowered {

std::vector<RuntimeConfig::LoopDescriptor>::iterator RuntimeConfig::get_loop_desc_it(size_t loop_id, RuntimeConfig::LoopDescriptor::Type type) {
    OPENVINO_ASSERT(loops.count(loop_id) > 0, "LoopId has not been found!");
    auto& loop_descriptors = loops.at(loop_id);
    return std::find_if(loop_descriptors.begin(), loop_descriptors.end(),
                        [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
}

bool RuntimeConfig::get_loop_desc(size_t loop_id, LoopDescriptor::Type type, LoopDescriptor& desc, size_t& index) const {
    OPENVINO_ASSERT(loops.count(loop_id) > 0, "LoopID has not been found!");
    index = 0;
    for (const auto& p : loops) {
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

bool RuntimeConfig::get_loop_desc(size_t loop_id, LoopDescriptor::Type type, LoopDescriptor& desc) const {
    OPENVINO_ASSERT(loops.count(loop_id) > 0, "LoopId has not been found!");
    const auto& loop_descriptors = loops.at(loop_id);
    const auto desc_it = std::find_if(loop_descriptors.cbegin(), loop_descriptors.cend(),
                                      [&type](const RuntimeConfig::LoopDescriptor& desc) { return desc.type == type; });
    if (desc_it != loop_descriptors.cend()) {
        desc = *desc_it;
        return true;
    }
    return false;
}

} // namespace lowered
} // namespace snippets
} // namespace ov
