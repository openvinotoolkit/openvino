// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <list>
#include <vector>
#include <algorithm>
#include <functional>

namespace GNAPluginNS {

enum rType {
    REQUEST_STORE,
    REQUEST_ALLOCATE,
    REQUEST_BIND,
    REQUEST_INITIALIZER,
};
/**
 * @brief region of firmware data
 */
enum rRegion {
    REGION_RO,
    REGION_RW,
    REGION_AUTO,
};

struct MemRequest {
    rType _type;
    rRegion  _region;
    void *_ptr_out;
    const void *_ptr_in = nullptr;
    std::function<void(void * data, size_t size)> _initializer;
    // holds arbitrary value
    std::vector<uint8_t> _data;
    uint8_t _element_size;
    size_t _num_elements;
    size_t _alignment;
    size_t _offset;
    // expansion in bytes due to large depended layers
    size_t _padding = 0;
    MemRequest(rRegion region,
                rType req,
                void *ptr_out,
                const void *ptr_in,
                uint8_t element_size = 0,
                size_t num_elements = 0,
                size_t alignment = 1,
                size_t offset = 0) : _region(region),
                                     _type(req),
                                     _ptr_out(ptr_out),
                                     _ptr_in(ptr_in),
                                     _element_size(element_size),
                                     _num_elements(num_elements),
                                     _alignment(alignment),
                                     _offset(offset) {}

    /**
     * Store value only request
     * @tparam T
     * @param req
     * @param ptr_out
     * @param element
     * @param num_elements
     * @param alignment
     */
    template<class T>
    MemRequest(rRegion region,
                void *ptr_out,
                T element,
                size_t num_elements,
                size_t alignment = 1) : _region(region),
                                        _type(REQUEST_STORE),
                                        _ptr_out(ptr_out),
                                        _element_size(sizeof(T)),
                                        _num_elements(num_elements),
                                        _alignment(alignment) {
        _data.resize(sizeof(T));
        std::copy(reinterpret_cast<uint8_t *>(&element), reinterpret_cast<uint8_t *>(&element) + sizeof(T), _data.begin());
    }
/**
     * Store initializer request
     * @param req
     * @param ptr_out
     * @param element
     * @param num_elements
     * @param alignment
     */
    MemRequest(rRegion region,
               void   *ptr_out,
               size_t  regionSize,
               std::function<void(void * data, size_t size)> initializer,
               size_t  alignment = 1) : _region(region),
                                        _type(REQUEST_INITIALIZER),
                                        _ptr_out(ptr_out),
                                        _element_size(1),
                                        _num_elements(regionSize),
                                        _alignment(alignment),
                                        _initializer(initializer) {
    }
};

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
        futureHeap().push_back({regionType(), ptr_out, num_bytes, initializer, alignment});
    }

    void push_ptr(void *ptr_out, const void *ptr_in, size_t num_bytes, size_t alignment = 1) {
        futureHeap().push_back({regionType(), REQUEST_STORE, ptr_out, ptr_in, 1, num_bytes, alignment});
    }

    /**
     * copy input to intermediate buffer
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
    void reserve_ptr(void *ptr_out, size_t num_bytes)  {
        futureHeap().push_back({regionType(), REQUEST_ALLOCATE, ptr_out, nullptr, 1, num_bytes});
    }

    /**
     *
     * @param source
     * @param dest - source is binded to dest pointer after allocation
     * @param offset - offset in bytes in sourse that will be set in dest
     * @param num_bytes - bind can request for bigger buffer that originally allocated via reserve(),
     *      if that happens - reserved request parameters will be updated bero commiting memory
     */
    void bind_ptr(void *source, const void *dest, size_t offset = 0, size_t num_bytes = 0)  {
        futureHeap().push_back({regionType(), REQUEST_BIND, source, dest, 1, num_bytes, 1, offset});
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



}  // namespace GNAPluginNS