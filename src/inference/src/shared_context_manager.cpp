// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_context_manager.hpp"

#include "openvino/runtime/icache_manager.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {

void SharedContextManager::clean() {
    std::lock_guard lock(m_mutex);
    erase_expired();
}

void SharedContextManager::init_and_sync_context(uint64_t cache_id, ov::IContextStore* context_store) {
    std::lock_guard lock(m_mutex);
    erase_expired();
    if (context_store) {
        context_store->initialize(get_context(cache_id).lock());
        set_context(cache_id, context_store->get_context());
    }
}

std::weak_ptr<ov::wsh::Context> SharedContextManager::get_context(uint64_t id) const {
    const auto ctx_it = m_weight_contexts.find(id);
    return ctx_it != m_weight_contexts.end() ? ctx_it->second : std::weak_ptr<ov::wsh::Context>{};
}

void SharedContextManager::set_context(uint64_t id, std::weak_ptr<ov::wsh::Context> context) {
    if (!context.expired()) {
        m_weight_contexts[id] = std::move(context);
    }
}

void SharedContextManager::erase_expired() {
    util::erase_if(m_weight_contexts, [](const auto& ctx_entry) {
        return ctx_entry.second.expired();
    });
}
}  // namespace ov
