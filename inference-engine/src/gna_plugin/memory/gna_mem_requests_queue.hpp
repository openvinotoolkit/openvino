// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <vector>
#include <algorithm>
#include <functional>
#include "gna_mem_requests.hpp"

namespace GNAPluginNS {
namespace memory {
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
    void push_initializer(void *ptr_out, size_t num_bytes, std::function<void(void * data, size_t size)> initializer, size_t alignment = 1) {
        futureHeap().push_back({regionType(), ptr_out, num_bytes, initializer, REQUEST_INITIALIZER, alignment});
    }

    void push_ptr(void *ptr_out, const void *ptr_in, size_t num_bytes, size_t alignment = 1) {
        futureHeap().push_back({regionType(), REQUEST_STORE, ptr_out, ptr_in, 1, num_bytes, alignment});
    }

    /**
     * @brief copy input to intermediate buffer, to further use in copying into gna-blob
     * @param ptr_out
     * @param ptr_in
     * @param num_bytes
     */
    void push_local_ptr(void *ptr_out, const void *ptr_in, size_t num_bytes, size_t alignment = 1) {
        localStorage().emplace_back(reinterpret_cast<const uint8_t *>(ptr_in),
                                    reinterpret_cast<const uint8_t *>(ptr_in) + num_bytes);
        futureHeap().push_back({regionType(), REQUEST_STORE, ptr_out, &localStorage().back().front(), 1, num_bytes, alignment});
    }

    /**
     *
     * @param ptr_out
     * @param num_bytes
     */
    void reserve_ptr(void *ptr_out, size_t num_bytes, size_t alignment = 1)  {
        futureHeap().push_back({regionType(), REQUEST_ALLOCATE, ptr_out, nullptr, 1, num_bytes, alignment});
    }

    /**
     *
     * @param source
     * @param dest - source is binded to dest pointer after allocation
     * @param offset - offset in bytes in source that will be set in dest
     * @param num_bytes - bind can request for bigger buffer that originally allocated via reserve(),
     *      if that happens - reserved request parameters will be updated before committing memory
     */
    void bind_ptr(void *source, const void *dest, size_t offset = 0, size_t num_bytes = 0)  {
        futureHeap().push_back({regionType(), REQUEST_BIND, source, dest, 1, num_bytes, 1, offset});
    }

    /**
     * @brief allows initialisation of previously requested segment, ex. const input of concat layer
     * @param ptr_out - previously requested buffer
     * @param initializer - initialisation routine to be called on allocated memory
     */
    void bind_initializer(void *ptr_out, std::function<void(void * data, size_t size)> initializer)  {
        futureHeap().push_back({regionType(), ptr_out, 0, initializer, REQUEST_BIND, 1});
    }

    /**
     * @brief allocates buffer and set all its values to T value
     */
    template<class T>
    void push_value(void *ptr_out, T value, size_t num_elements, size_t alignment = 1) {
        futureHeap().push_back({regionType(), ptr_out, value, num_elements, alignment});
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
