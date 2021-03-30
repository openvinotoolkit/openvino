// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <mutex>
#include <memory>
#include <atomic>
#include <unordered_map>

namespace InferenceEngine {

class CacheGuard;
class CacheGuardEntry {
public:
    CacheGuardEntry(CacheGuard& cacheGuard, const std::string& hash,
                    std::shared_ptr<std::mutex> m, std::atomic_int& refCount);
    CacheGuardEntry(const CacheGuardEntry&) = delete;

    ~CacheGuardEntry();

    // Will be called by CacheGuard to perform real lock
    void performLock();
private:
    CacheGuard& m_cacheGuard;
    std::string m_hash;
    std::shared_ptr<std::mutex> m_mutex;
    std::atomic_int& m_refCount;
};

class CacheGuard {
public:
    CacheGuard() = default;

    std::unique_ptr<CacheGuardEntry> getHashLock(const std::string& hash);
    void checkForRemove(const std::string& hash);
private:
    struct Item {
        std::shared_ptr<std::mutex> m_mutexPtr { std::make_shared<std::mutex>() };
        // RefCount for item usage
        std::atomic_int m_itemRefCounter {0};

        Item() = default;
        Item(Item&& other): m_mutexPtr(std::move(other.m_mutexPtr)),
                            m_itemRefCounter(other.m_itemRefCounter.load()) {}
    };
    std::mutex m_tableMutex;
    std::unordered_map<std::string, Item> m_table;
};

}  // namespace InferenceEngine
