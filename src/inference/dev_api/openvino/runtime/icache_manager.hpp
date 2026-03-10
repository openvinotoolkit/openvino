// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <clocale>
#include <functional>
#include <iostream>
#include <string>
#include <variant>

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

/**
 * @brief This class represents private interface for Cache Manager
 */
class ICacheManager {
public:
    /**
     * @brief Default destructor
     */
    virtual ~ICacheManager() = default;

    /**
     * @brief Function passing created output stream
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
 * @brief Interface to store and get shared context.
 */
class IContextStore {
public:
    /**
     * @brief Writes context to the storage
     * @param context The context to be stored
     */
    virtual void write_context(const ov::weight_sharing::Context&) = 0;

    /**
     * @brief Gets context from the storage
     * @return The stored context
     */
    virtual ov::weight_sharing::Context get_context() const = 0;
};

/**
 * @brief This class limits the locale env to a special value in sub-scope
 */
class ScopedLocale {
public:
    ScopedLocale(int category, std::string new_locale) : m_category(category) {
        m_old_locale = setlocale(category, nullptr);
        setlocale(m_category, new_locale.c_str());
    }
    ~ScopedLocale() {
        setlocale(m_category, m_old_locale.c_str());
    }

    // Disable heap allocation - it's meant for local scope usage only.
    static void* operator new(std::size_t) = delete;
    static void* operator new[](std::size_t) = delete;

private:
    int m_category;
    std::string m_old_locale;
};

}  // namespace ov
