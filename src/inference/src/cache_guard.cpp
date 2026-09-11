// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_guard.hpp"

namespace ov {

CacheGuardEntry::CacheGuardEntry(CacheGuard& cacheGuard,
                                 const std::string& hash,
                                 std::shared_ptr<std::mutex> m,
                                 std::atomic_int& refCount)
    : m_cacheGuard(cacheGuard),
      m_hash(hash),
      m_mutex(std::move(m)),
      m_refCount(refCount),
      // Deferred: don't lock the mutex here for exception-safety considerations.
      m_lock(*m_mutex, std::defer_lock) {
    m_refCount++;
}

CacheGuardEntry::~CacheGuardEntry() {
    m_refCount--;
    // Unlock only if this entry actually acquired the lock. perform_lock() is
    // called after construction, so the entry may be destroyed (e.g. on an
    // exception path) without ever having locked the mutex.
    if (m_lock.owns_lock()) {
        m_lock.unlock();
    }
    m_cacheGuard.check_for_remove(m_hash);
}

void CacheGuardEntry::perform_lock() {
    m_lock.lock();
}

//////////////////////////////////////////////////////

std::unique_ptr<CacheGuardEntry> CacheGuard::get_hash_lock(const std::string& hash) {
    std::unique_ptr<CacheGuardEntry> res;
    {
        std::unique_lock<std::mutex> lock(m_tableMutex);
        auto& data = m_table[hash];
        try {
            // TODO: use std::make_unique when migrated to C++14
            res = std::unique_ptr<CacheGuardEntry>(
                new CacheGuardEntry(*this, hash, data.m_mutexPtr, data.m_itemRefCounter));
        } catch (...) {
            // In case of exception, we shall remove hash entry if it is not used
            if (data.m_itemRefCounter == 0) {
                m_table.erase(hash);
            }
            throw;
        }
    }
    res->perform_lock();  // in case of exception, 'res' will be destroyed and item will be cleaned up from table
    return res;
}

void CacheGuard::check_for_remove(const std::string& hash) {
    std::lock_guard<std::mutex> lock(m_tableMutex);
    if (m_table.count(hash)) {
        auto& data = m_table[hash];
        if (data.m_itemRefCounter == 0) {
            // Nobody is using this and nobody is waiting for it - can be removed
            m_table.erase(hash);
        }
    }
}

}  // namespace ov
