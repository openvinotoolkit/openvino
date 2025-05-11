// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief This is a header file for the OpenVINO Cache Guard class C++ API
 *
 * @file cache_guard.hpp
 */

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ov {

class CacheGuard;
/**
 * @brief This class represents RAII guard class to protect multiple threads to modify the same cached network
 * Use CacheGuard::getHashLock(hash) to acquire lock for specific cache entry identified by its 'hash'
 * On destruction, lock will be released
 * @see CacheGuard
 */
class CacheGuardEntry {
public:
    /**
     * @brief Internal constructor, will be called by @CacheGuard
     *
     * @param cacheGuard Reference link to parent's Cache Guard
     * @param hash String representing hash of network
     * @param m Shared pointer to mutex for internal locking
     * @param refCount Reference counter. Will be decremented on CacheGuardEntry destruction
     */
    CacheGuardEntry(CacheGuard& cacheGuard,
                    const std::string& hash,
                    std::shared_ptr<std::mutex> m,
                    std::atomic_int& refCount);
    CacheGuardEntry(const CacheGuardEntry&) = delete;
    CacheGuardEntry& operator=(const CacheGuardEntry&) = delete;

    /**
     * @brief Destructor, will perform the following cleanup
     *
     * Decrement reference counter
     * Unlock associated mutex
     * Call CacheGuard::checkForRemove to check if appropriate table hash entry is not used anymore and can be deleted
     */
    ~CacheGuardEntry();

    /**
     * @brief Performs real lock of associated mutex
     * It is separated from construction due to exception safety considerations
     *
     * @note Will be called only by CacheGuard, it shall not be called from client's code
     */
    void perform_lock();

private:
    CacheGuard& m_cacheGuard;
    std::string m_hash;
    std::shared_ptr<std::mutex> m_mutex;
    std::atomic_int& m_refCount;
};

/**
 * @brief This class holds a table of currently locked hashes
 * OpenVINO core will need to obtain a lock for a specific cache to get exclusive access to it
 * It is needed to avoid race situations when multiple threads try to to write to the same cache simultaneously
 *
 * Usage example:
 *     auto hash = <calculate hash for network>;
 *     {
 *         auto lock = m_cacheGuard.getHashLock(hash);
 *         <work with cache entry exclusively>
 *     }
 */
class CacheGuard {
public:
    CacheGuard() = default;

    /**
     * @brief Gets a lock for a specific cache entry identified by it's hash value
     * Once returned, client has an exclusive access to cache entry for read/write/delete
     * If any other thread holds a lock to same hash - this function will not return until it is unlocked
     *
     * @param hash String representing hash of network
     *
     * @return RAII pointer to CacheGuardEntry
     */
    std::unique_ptr<CacheGuardEntry> get_hash_lock(const std::string& hash);

    /**
     * @brief Checks whether there is any clients holding the lock after CacheGuardEntry deletion
     * It will be called on destruction of CacheGuardEntry and shall not be used directly by client's code
     * If there is no more clients holding the lock, associated entry will be removed from table unlocked
     *
     * @param hash String representing hash of network
     *
     * @return RAII pointer to CacheGuardEntry
     */
    void check_for_remove(const std::string& hash);

private:
    struct Item {
        std::shared_ptr<std::mutex> m_mutexPtr{std::make_shared<std::mutex>()};
        // Reference counter for item usage
        std::atomic_int m_itemRefCounter{0};

        Item() = default;
        Item(const Item& other) : m_mutexPtr(other.m_mutexPtr), m_itemRefCounter(other.m_itemRefCounter.load()) {}
        Item& operator=(const Item& other) = delete;
        Item(Item&& other) : m_mutexPtr(std::move(other.m_mutexPtr)), m_itemRefCounter(other.m_itemRefCounter.load()) {}
        Item& operator=(Item&& other) = delete;
    };
    std::mutex m_tableMutex;
    std::unordered_map<std::string, Item> m_table;
};

}  // namespace ov
