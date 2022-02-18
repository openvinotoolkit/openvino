// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <vector>
#include <algorithm>
#include <functional>

#include <ie_api.h>
#include <legacy/ie_layers.h>
#include "gna_mem_requests.hpp"

namespace GNAPluginNS {
namespace memory {

/**
* @brief get layer id from legacy CNNLayer
*/
inline uint16_t getCNNLayerId(InferenceEngine::CNNLayerPtr layer) {
    IE_SUPPRESS_DEPRECATED_START
    return layer->userValue.v_int;
    IE_SUPPRESS_DEPRECATED_END
}

/**
 * Adapter for requests submission and actual request queue
 */
class GNAMemRequestsQueue {
public:
    virtual ~GNAMemRequestsQueue() {}

    /**
     * @brief register initialiser to access memory once it is actually allocated
     * @param ptr_out
     * @param ptr_in
     * @param num_bytes
     * @param alignment
     */
    void push_initializer(InferenceEngine::CNNLayerPtr layer,
                          void *ptr_out,
                          size_t num_bytes,
                          std::function<void(void * data, size_t size)> initializer,
                          size_t alignment = 1) {
        futureHeap().push_back({regionType(), ptr_out, num_bytes, initializer, REQUEST_INITIALIZER, alignment});
        if (layer != nullptr) {
            futureHeap().back()._life_limits = {0, getCNNLayerId(layer)};
        }
    }

    void push_ptr(InferenceEngine::CNNLayerPtr layer,
                  void *ptr_out,
                  const void *ptr_in,
                  size_t num_bytes,
                  size_t alignment = 1) {
        futureHeap().push_back({regionType(), REQUEST_STORE, ptr_out, ptr_in, 1, num_bytes, alignment});
        if (layer != nullptr) {
            futureHeap().back()._life_limits = {0, getCNNLayerId(layer)};
        }
    }

    /**
     * @brief copy input to intermediate buffer, to further use in copying into gna-blob
     * @param ptr_out
     * @param ptr_in
     * @param num_bytes
     */
    void push_local_ptr(InferenceEngine::CNNLayerPtr layer,
                        void *ptr_out,
                        const void *ptr_in,
                        size_t num_bytes,
                        size_t alignment = 1) {
        localStorage().emplace_back(reinterpret_cast<const uint8_t *>(ptr_in),
                                    reinterpret_cast<const uint8_t *>(ptr_in) + num_bytes);
        futureHeap().push_back({regionType(), REQUEST_STORE, ptr_out, &localStorage().back().front(), 1, num_bytes, alignment});
        if (layer != nullptr) {
            futureHeap().back()._life_limits = {0, getCNNLayerId(layer)};
        }
    }

    /**
     *
     * @param ptr_out
     * @param num_bytes
     */
    void reserve_ptr(InferenceEngine::CNNLayerPtr layer,
                     void *ptr_out,
                     size_t num_bytes,
                     size_t alignment = 1)  {
        futureHeap().push_back({regionType(), REQUEST_ALLOCATE, ptr_out, nullptr, 1, num_bytes, alignment});
        if (layer != nullptr) {
            futureHeap().back()._life_limits = {getCNNLayerId(layer), getCNNLayerId(layer)};
        }
    }

    /**
     *
     * @param source
     * @param dest - source is binded to dest pointer after allocation
     * @param offset - offset in bytes in source that will be set in dest
     * @param num_bytes - bind can request for bigger buffer that originally allocated via reserve(),
     *      if that happens - reserved request parameters will be updated before committing memory
     */
    void bind_ptr(InferenceEngine::CNNLayerPtr layer,
                  void *source,
                  const void *dest,
                  size_t offset = 0,
                  size_t num_bytes = 0)  {
        futureHeap().push_back({regionType(), REQUEST_BIND, source, dest, 1, num_bytes, 1, offset});
        if (layer != nullptr) {
            futureHeap().back()._life_limits = {getCNNLayerId(layer), getCNNLayerId(layer)};
        }
    }

    /**
     * @brief allows initialisation of previously requested segment, ex. const input of concat layer
     * @param ptr_out - previously requested buffer
     * @param initializer - initialisation routine to be called on allocated memory
     */
    void bind_initializer(InferenceEngine::CNNLayerPtr layer,
                          void *ptr_out,
                          std::function<void(void * data, size_t size)> initializer) {
        futureHeap().push_back({regionType(), ptr_out, 0, initializer, REQUEST_BIND, 1});
        if (layer != nullptr) {
            futureHeap().back()._life_limits = {0, getCNNLayerId(layer)};
        }
    }

    /**
     * @brief allocates buffer and set all its values to T value
     */
    template<class T>
    void push_value(InferenceEngine::CNNLayerPtr layer,
                    void *ptr_out,
                    T value,
                    size_t num_elements,
                    size_t alignment = 1) {
        futureHeap().push_back({regionType(), ptr_out, value, num_elements, alignment});
        if (layer != nullptr) {
            futureHeap().back()._life_limits = {0, getCNNLayerId(layer)};
        }
    }

    /**
     * @brief interface for actual queue storage
     */
    virtual rRegion regionType() const = 0;
    virtual std::vector<MemRequest> & futureHeap()  = 0;
    virtual std::list<std::vector<char>> &localStorage() = 0;
};
}  // namespace memory
}  // namespace GNAPluginNS
