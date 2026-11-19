// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ov {
namespace tp {

/// Process-wide locks of the devices of a TP group, held for a whole TP request.
///
/// TP requests of compiled models whose device groups overlap must not run
/// their collectives concurrently: a device queue waiting for a peer could hold
/// back the other model's work on that device while the opposite happens on
/// another one. Every group takes its devices' locks in one global order (by
/// key), so partly overlapping groups cannot deadlock on each other.
class DeviceLocks {
public:
    /// \param keys One key per device that identifies it across aliases (its UUID).
    explicit DeviceLocks(std::vector<std::string> keys) {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        std::lock_guard<std::mutex> lock(registry_mutex());
        auto& registry = this->registry();
        for (const auto& key : keys) {
            auto& entry = registry[key];
            auto mutex = entry.lock();
            if (!mutex) {
                mutex = std::make_shared<std::mutex>();
                entry = mutex;
            }
            m_mutexes.push_back(std::move(mutex));
        }
    }

    /// Blocks until every device of the group is free; released when the result is destroyed.
    std::vector<std::unique_lock<std::mutex>> lock() const {
        std::vector<std::unique_lock<std::mutex>> held;
        held.reserve(m_mutexes.size());
        for (const auto& mutex : m_mutexes) {
            held.emplace_back(*mutex);
        }
        return held;
    }

private:
    using Registry = std::map<std::string, std::weak_ptr<std::mutex>>;

    static std::mutex& registry_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    static Registry& registry() {
        static Registry registry;
        return registry;
    }

    std::vector<std::shared_ptr<std::mutex>> m_mutexes;
};

}  // namespace tp
}  // namespace ov
