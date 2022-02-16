// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_cache_guard.hpp"

#include "ie_common.h"

namespace InferenceEngine {

CacheGuardEntry::CacheGuardEntry(CacheGuard& cacheGuard,
                                 const std::string& hash,
                                 std::shared_ptr<std::mutex> m,
                                 std::atomic_int& refCount)
    : m_cacheGuard(cacheGuard),
      m_hash(hash),
      m_mutex(m),
      m_refCount(refCount) {
    // Don't lock mutex right here for exception-safe considerations
    m_refCount++;
}

CacheGuardEntry::~CacheGuardEntry() {
    m_refCount--;
    m_mutex->unlock();
    m_cacheGuard.checkForRemove(m_hash);
}

void CacheGuardEntry::performLock() {
    m_mutex->lock();
}

//////////////////////////////////////////////////////

std::unique_ptr<CacheGuardEntry> CacheGuard::getHashLock(const std::string& hash) {
    std::unique_lock<std::mutex> lock(m_tableMutex);
    auto& data = m_table[hash];
    std::unique_ptr<CacheGuardEntry> res;
    try {
        // TODO: use std::make_unique when migrated to C++14
        res =
            std::unique_ptr<CacheGuardEntry>(new CacheGuardEntry(*this, hash, data.m_mutexPtr, data.m_itemRefCounter));
    } catch (...) {
        // In case of exception, we shall remove hash entry if it is not used
        if (data.m_itemRefCounter == 0) {
            m_table.erase(hash);
        }
        throw;
    }
    lock.unlock();       // can unlock table lock here, as refCounter is positive and nobody can remove entry
    res->performLock();  // in case of exception, 'res' will be destroyed and item will be cleaned up from table
    return res;
}

void CacheGuard::checkForRemove(const std::string& hash) {
    std::lock_guard<std::mutex> lock(m_tableMutex);
    if (m_table.count(hash)) {
        auto& data = m_table[hash];
        if (data.m_itemRefCounter == 0) {
            // Nobody is using this and nobody is waiting for it - can be removed
            m_table.erase(hash);
        }
    }
}

}  // namespace InferenceEngine
