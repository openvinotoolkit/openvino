// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <variant>

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/tensor.hpp"

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

class ISharedContextStore {
public:
    virtual void write_context_entry(const SharedContext&) = 0;
    virtual SharedContext get_shared_context() const = 0;
};

}  // namespace ov
