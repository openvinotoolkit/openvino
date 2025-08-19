// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Cache Manager class C++ API
 *
 * @file cache_manager.hpp
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

/**
 * @brief This class limits the locale env to a special value in sub-scope
 *
 */
class ScopedLocale {
public:
    ScopedLocale(int category, std::string newLocale) : m_category(category) {
        m_oldLocale = setlocale(category, nullptr);
        setlocale(m_category, newLocale.c_str());
    }
    ~ScopedLocale() {
        setlocale(m_category, m_oldLocale.c_str());
    }

private:
    int m_category;
    std::string m_oldLocale;
};

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
     * @brief Variant type for compiled blob representation
     */
    using CompiledBlobVariant = std::variant<const ov::Tensor, std::reference_wrapper<std::istream>>;

    /**
     * @brief Function passing created input stream
     */
    using StreamReader = std::function<void(CompiledBlobVariant&)>;

    /**
     * @brief Callback when OpenVINO intends to read model from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, model will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the model)
     * @param enable_mmap use mmap or ifstream to read model file
     * @param reader Lambda function to be called when input stream is created
     */
    virtual void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) = 0;

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

    ov::util::Path getBlobFile(const std::string& blobHash) const {
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
        return ov::util::string_to_wstring(ov::util::make_path(m_cachePath, blobHash + ".blob"));
#else
        return ov::util::make_path(m_cachePath, blobHash + ".blob");
#endif
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
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        std::ofstream stream(getBlobFile(id), std::ios_base::binary | std::ofstream::out);
        writer(stream);
    }

    void read_cache_entry(const std::string& id, bool enable_mmap, StreamReader reader) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        const auto blob_file_name = getBlobFile(id);
        if (std::filesystem::exists(blob_file_name)) {
            if (enable_mmap) {
                CompiledBlobVariant compiled_blob{std::in_place_index<0>, ov::read_tensor_data(blob_file_name)};
                reader(compiled_blob);
            } else {
                std::ifstream stream(blob_file_name, std::ios_base::binary);
                CompiledBlobVariant compiled_blob{std::in_place_index<1>, std::ref(stream)};
                reader(compiled_blob);
            }
        }
    }

    void remove_cache_entry(const std::string& id) override {
        auto blobFileName = getBlobFile(id);

        if (std::filesystem::exists(blobFileName)) {
            std::ignore = std::filesystem::remove(blobFileName);
        }
    }
};

}  // namespace ov
