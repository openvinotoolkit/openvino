// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Cache Manager class C++ API
 *
 * @file cache_manager.hpp
 */
#pragma once

#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include "openvino/util/file_util.hpp"

namespace ov {

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
     * @brief Callback when OpenVINO intends to write model to cache
     *
     * Client needs to call create std::ostream object and call writer(ostream)
     * Otherwise, model will not be cached
     *
     * @param id Id of cache (hash of the model)
     * @param writer Lambda function to be called when stream is created
     */
    virtual void write_cache_entry(const std::string& id, StreamWriter writer) = 0;

    /**
     * @brief Function passing created input stream
     */
    using StreamReader = std::function<void(std::istream&)>;

    /**
     * @brief Callback when OpenVINO intends to read model from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, model will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the model)
     * @param reader Lambda function to be called when input stream is created
     */
    virtual void read_cache_entry(const std::string& id, StreamReader reader) = 0;

    /**
     * @brief Callback when OpenVINO intends to remove cache entry
     *
     * Client needs to perform appropriate cleanup (e.g. delete a cache file)
     *
     * @param id Id of cache (hash of the model)
     */
    virtual void remove_cache_entry(const std::string& id) = 0;
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
        return ov::util::make_path(m_cachePath, blobHash + ".blob");
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
    void write_cache_entry(const std::string& id, StreamWriter writer) override {
        std::ofstream stream(getBlobFile(id), std::ios_base::binary | std::ofstream::out);
        writer(stream);
    }

    void read_cache_entry(const std::string& id, StreamReader reader) override {
        auto blobFileName = getBlobFile(id);
        if (ov::util::file_exists(blobFileName)) {
            std::ifstream stream(blobFileName, std::ios_base::binary);
            reader(stream);
        }
    }

    void remove_cache_entry(const std::string& id) override {
        auto blobFileName = getBlobFile(id);
        if (ov::util::file_exists(blobFileName))
            std::remove(blobFileName.c_str());
    }
};

}  // namespace ov
