// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "openvino/core/weight_sharing_util.hpp"

namespace ov {

class IContextStore;

class SharedContextManager {
public:
    /**
     * @brief Initialize shared context in context store object and synchronize it with shared context manager.
     *
     * @param cache_id Cache identifier.
     * @param context_store Pointer to shared context store object.
     */
    void init_and_sync_context(uint64_t cache_id, ov::IContextStore* context_store);

    /** @brief Clean up Shared Context Manager by removing expired contexts. */
    void clean();

private:
    std::weak_ptr<ov::wsh::Context> get_context(uint64_t id) const;
    void set_context(uint64_t id, std::weak_ptr<ov::wsh::Context> context);
    void erase_expired();

    mutable std::mutex m_mutex{};
    std::unordered_map<uint64_t, std::weak_ptr<ov::wsh::Context>> m_weight_contexts{};
};
}  // namespace ov
