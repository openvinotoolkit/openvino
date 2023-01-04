// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the Inference Engine Cache Manager class C++ API
 *
 * @file ie_cache_manager.hpp
 */
#pragma once

#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include "file_utils.h"
#include "ie_api.h"

namespace InferenceEngine {

/**
 * @brief This class represents private interface for Cache Manager
 *
 */
class ICacheManager {
public:
    /**
     * @brief Default destructor
     */
    virtual ~ICacheManager() = default;

    /**
     * @brief Function passing created output stream
     *
     */
    using StreamWriter = std::function<void(std::ostream&)>;
    /**
     * @brief Callback when Inference Engine intends to write network to cache
     *
     * Client needs to call create std::ostream object and call writer(ostream)
     * Otherwise, network will not be cached
     *
     * @param id Id of cache (hash of the network)
     * @param writer Lambda function to be called when stream is created
     */
    virtual void writeCacheEntry(const std::string& id, StreamWriter writer) = 0;

    /**
     * @brief Function passing created input stream
     *
     */
    using StreamReader = std::function<void(std::istream&)>;
    /**
     * @brief Callback when Inference Engine intends to read network from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, network will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the network)
     * @param reader Lambda function to be called when input stream is created
     */
    virtual void readCacheEntry(const std::string& id, StreamReader reader) = 0;

    /**
     * @brief Callback when Inference Engine intends to remove cache entry
     *
     * Client needs to perform appropriate cleanup (e.g. delete a cache file)
     *
     * @param id Id of cache (hash of the network)
     */
    virtual void removeCacheEntry(const std::string& id) = 0;
};

/**
 * @brief File storage-based Implementation of ICacheManager
 *
 * Uses simple file for read/write cached models.
 *
 */
class FileStorageCacheManager final : public ICacheManager {
    std::string m_cachePath;

    std::string getBlobFile(const std::string& blobHash) const {
        return FileUtils::makePath(m_cachePath, blobHash + ".blob");
    }

public:
    /**
     * @brief Constructor
     *
     */
    FileStorageCacheManager(std::string cachePath) : m_cachePath(std::move(cachePath)) {}

    /**
     * @brief Destructor
     *
     */
    ~FileStorageCacheManager() override = default;

private:
    void writeCacheEntry(const std::string& id, StreamWriter writer) override {
        std::ofstream stream(getBlobFile(id), std::ios_base::binary | std::ofstream::out);
        writer(stream);
    }

    void readCacheEntry(const std::string& id, StreamReader reader) override {
        auto blobFileName = getBlobFile(id);
        if (FileUtils::fileExist(blobFileName)) {
            std::ifstream stream(blobFileName, std::ios_base::binary);
            reader(stream);
        }
    }

    void removeCacheEntry(const std::string& id) override {
        auto blobFileName = getBlobFile(id);
        if (FileUtils::fileExist(blobFileName))
            std::remove(blobFileName.c_str());
    }
};

}  // namespace InferenceEngine
