// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "gna_mem_requests.hpp"
#include "ie_memcpy.h"
#include "gna_mem_requests_queue.hpp"
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <functional>

/**
 * Pads memory size to given number of Bytes
 *
 * Please always use this padding macro for consistency
 *
 * @memSize size (in bytes) of memory to be padded
 * @align   number of bytes to pad
 * @return  memory size (int bytes) padded to given value
 */
#ifndef ALIGN
# define ALIGN(memSize, pad)   (static_cast<int>(((memSize) + pad -1) / pad) * pad)
#endif

namespace GNAPluginNS {
namespace memory {
/**
 * @brief encapsulate various request to allocate GNA specific memory,
 * in order to issue single allocation call and configure actual pointers in requests
 * @tparam Allocator - a GNAAllocator in case of actual HW offloads
 */
template<class Allocator = std::allocator<uint8_t>>
class GNAMemory : public GNAMemRequestsQueue {
    std::vector<MemRequest> _future_heap;
    std::list<std::vector<char>> _local_storage;
    size_t _total = 0;
    size_t _rw_section_size = 0;
    size_t _ro_section_size = 0;
    Allocator _allocator;
    std::shared_ptr<uint8_t> heap;
    size_t _page_alignment = 1;

    class GNAMemRequestsReadOnlyQueue : public GNAMemRequestsQueue {
        std::reference_wrapper<GNAMemRequestsQueue> _that;
     public:
        explicit GNAMemRequestsReadOnlyQueue(GNAMemory & that) : _that(that) {
        }
        rRegion regionType() const override {
            return REGION_RO;
        };
        std::vector<MemRequest> & futureHeap()  override {
            return _that.get().futureHeap();
        }
        std::list<std::vector<char>> &localStorage() override {
            return _that.get().localStorage();
        }
    };

    GNAMemRequestsReadOnlyQueue readOnlyFrontEnd;

 public:
    explicit GNAMemory(size_t pageAlignment = 1)
        : readOnlyFrontEnd(*this), _page_alignment(pageAlignment) {}

    explicit GNAMemory(const Allocator &a, size_t pageAlignment = 1)
        : _allocator(a), readOnlyFrontEnd(*this), _page_alignment(pageAlignment) {}

    GNAMemRequestsQueue & readonly() {
        return readOnlyFrontEnd;
    }

    /**
     * @brief calculates size required for all requests, allocates memory and updates pointers
     */
    void commit() {
        // 1st stage -- looking for expandable bind requests:
        for (auto &originated : _future_heap) {
            if (originated._type & REQUEST_BIND) continue;
            size_t offset = 0;
            iterate_binded(originated, [&](MemRequest & reference, MemRequest & binded) {
                if (&originated == &reference) {
                    offset = 0;
                }
                offset += binded._offset;
                auto current = offset + ALIGN(binded._num_elements * binded._element_size, binded._alignment);
                auto original_no_pad = ALIGN(originated._num_elements * originated._element_size, originated._alignment);
                auto original_with_pad = ALIGN(originated._num_elements * originated._element_size + originated._padding, originated._alignment);

                originated._padding = ALIGN(std::max(original_with_pad, current), originated._alignment) - original_no_pad;
            });
        }

        updateSectionsSizes();

        _total = _rw_section_size + _ro_section_size;

        // allocation with memory setting to 0 internally
        heap = allocate(_total);
        auto setupOffsets = [&](std::function<bool(MemRequest & request)> filter, size_t offset) {
            for (auto &re : _future_heap) {
                if (re._type == REQUEST_BIND) continue;
                if (filter(re)) continue;

                auto sz = re._element_size * re._num_elements;

                if (re._ptr_out != nullptr) {
                    auto cptr = heap.get() + offset;
                    size_t cptr_avail_size = _total - offset;
                    if (re._type & REQUEST_BIND) {
                        cptr = reinterpret_cast<uint8_t*>(*reinterpret_cast<void **>(re._ptr_out));
                        cptr_avail_size = sz;
                    } else {
                        *reinterpret_cast<void **>(re._ptr_out) = cptr;
                    }
                    // std::cout << "ALLOCATED=" << cptr << ", size=" << re._element_size * re._num_elements << "\n";
                    iterate_binded(re, [](MemRequest & reference, MemRequest & binded) {
                        *reinterpret_cast<void **>(binded._ptr_out) =
                            binded._offset + reinterpret_cast<uint8_t *>(*reinterpret_cast<void **>(reference._ptr_out));
                        binded._num_elements = reference._num_elements;
                        binded._element_size = reference._element_size;
                    });

                    // std::cout << "size=" << ALIGN(sz, re._alignment) << "\n" << std::flush;

                    switch (re._type & ~REQUEST_BIND) {
                        case REQUEST_ALLOCATE :
                            break;
                        case REQUEST_STORE : {
                            if (re._ptr_in != nullptr) {
                                ie_memcpy(cptr, cptr_avail_size, re._ptr_in, sz);
                            } else {
                                size_t of = 0;
                                for (int i = 0; i < re._num_elements; i++, of += re._element_size) {
                                    std::copy(std::begin(re._data), std::end(re._data), cptr + of);
                                }
                            }
                            break;
                        }
                        case REQUEST_INITIALIZER : {
                            re._initializer(cptr, sz);
                            break;
                        }
                    }
                }
                if (!(re._type & REQUEST_BIND)) {
                    offset += ALIGN(sz + re._padding, re._alignment);
                }
            }
        };

        setupOffsets([](GNAPluginNS::memory::MemRequest & request) {
            // TODO: consume bind requests separately from storage type
            return !(request._type & REQUEST_BIND) && (request._region != REGION_RW);
        }, 0);

        setupOffsets([](GNAPluginNS::memory::MemRequest & request) {
            return (request._type & REQUEST_BIND) || request._region != REGION_RO;
        }, _rw_section_size);
    }

    void *getBasePtr() {
        return heap.get();
    }

    size_t getRWBytes() {
        updateSectionsSizes();
        return _rw_section_size;
    }

    size_t getTotalBytes() {
        updateSectionsSizes();
        return _total;
    }

 protected:
    rRegion regionType() const override {
        return REGION_RW;
    };
    std::vector<MemRequest> & futureHeap()  override {
        return _future_heap;
    }
    std::list<std::vector<char>> &localStorage() override {
        return _local_storage;
    }

    template<class T>
    void iterate_binded(GNAPluginNS::memory::MemRequest & reference, const T & visitor) {
        for (auto &re : _future_heap) {
            if ((re._type & REQUEST_BIND) && (re._ptr_in == reference._ptr_out)) {
                // std::cout << "  [binded=" << re._type << ", ptr=" << re._ptr_out <<"]\n";
                visitor(reference, re);
                // primitive loop check
                if (re._ptr_in == re._ptr_out) continue;
                // TODO: no circular dependency checking, only tree-style dependency with loops supported
                iterate_binded(re, visitor);
            }
        }
    }


    std::shared_ptr<uint8_t> allocate(size_t bytes) {
        std::shared_ptr<uint8_t> sp(_allocator.allocate(bytes), [=](uint8_t *p) {
            _allocator.deallocate(p, bytes);
        });
        std::fill(sp.get(), sp.get() + bytes, 0);
        return sp;
    }

 protected:
    void updateSectionsSizes() {
        // count total size and size of read/write regions
        _rw_section_size = 0;
        _ro_section_size = 0;
        for (auto &re : _future_heap) {
            auto current = ALIGN(re._num_elements * re._element_size + re._padding, re._alignment);
#ifdef GNA_HEAP_PROFILER
            std::cout << "chunk: " << " region: " << re._region << ", " <<
                    "type: " << (re._type  == REQUEST_STORE ? "store " : re._type == REQUEST_BIND ? "bind  " : "alloc ") <<
                    std::setw(10) << re._num_elements << ", " <<
                    static_cast<int>(re._element_size) << ", " <<
                    re._padding << ", " <<
                    re._offset << ", " <<
                    re._alignment << std::endl;
#endif
            if (re._type == REQUEST_BIND) continue;

            if (re._region == REGION_RW) {
                _rw_section_size += current;
            } else {
                _ro_section_size += current;
            }
        }
        _rw_section_size = ALIGN(_rw_section_size, _page_alignment);
        _ro_section_size = ALIGN(_ro_section_size, _page_alignment);
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
