// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna_mem_requests.hpp"
#include <ie_memcpy.h>
#include "gna_mem_requests_queue.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <functional>
#include <fstream>
#include <iomanip>
#include "gna_lib_ver_selector.hpp"
#include "gna_allocator.hpp"

namespace GNAPluginNS {
namespace memory {
/**
 * @brief encapsulate various request to allocate GNA specific memory,
 * in order to issue single allocation call and configure actual pointers in requests
 * @tparam Allocator - a GNAAllocator in case of actual HW offloads
 */
//template<class Allocator = std::allocator<uint8_t>>
    using Allocator = GNAAllocator;
class GNAMemory {
    std::map<rRegion, std::shared_ptr<GNAMemRequestsQueue>> _mem_queues;
    size_t _total = 0;
    Allocator _allocator;
    std::shared_ptr<uint8_t> heap = nullptr;
    size_t _page_alignment = 1;

 public:
    //explicit GNAMemory(size_t pageAlignment = 1)
    //    : _page_alignment(pageAlignment) {
    //        _mem_queues.insert(std::make_pair(REGION_INPUTS, new GNAMemRequestsInputsQueue));
    //        _mem_queues.insert(std::make_pair(REGION_OUTPUTS, new GNAMemRequestsOutputsQueue));
    //        _mem_queues.insert(std::make_pair(REGION_SCRATCH, new GNAMemRequestsScratchQueue));
    //        _mem_queues.insert(std::make_pair(REGION_RO, new GNAMemRequestsReadOnlyQueue));
    //        _mem_queues.insert(std::make_pair(REGION_STATES, new GNAMemRequestsStatesQueue));
    //        _mem_queues.insert(std::make_pair(REGION_AUTO, new GNAMemRequestsBindingsQueue));
    //    }

    explicit GNAMemory(const Allocator &a, size_t pageAlignment = 1)
        : _allocator(a), _page_alignment(pageAlignment) {
            _mem_queues.insert(std::make_pair(REGION_INPUTS, new GNAMemRequestsInputsQueue));
            _mem_queues.insert(std::make_pair(REGION_OUTPUTS, new GNAMemRequestsOutputsQueue));
            _mem_queues.insert(std::make_pair(REGION_SCRATCH, new GNAMemRequestsScratchQueue));
            _mem_queues.insert(std::make_pair(REGION_RO, new GNAMemRequestsReadOnlyQueue));
            _mem_queues.insert(std::make_pair(REGION_STATES, new GNAMemRequestsStatesQueue));
            _mem_queues.insert(std::make_pair(REGION_AUTO, new GNAMemRequestsBindingsQueue));
        }


    std::shared_ptr<GNAMemRequestsQueue> getQueue(rRegion region) {
        return _mem_queues[region];
    }

    void *getBasePtr() {
        return heap.get();
    }

    size_t getRWBytes() {
        return ALIGN(getQueue(REGION_STATES)->calcSize(), _page_alignment);
    }

    size_t getTotalBytes() {
        _total = 0;
        for (auto queue : _mem_queues) {
            expandBindRequests(queue.second);
            _total += ALIGN(queue.second->calcSize(), _page_alignment);
        }
        return _total;
    }

    size_t allocateRegion(std::shared_ptr<GNAMemRequestsQueue> mRequests) {
        size_t r_size = ALIGN(mRequests->getSize(), _page_alignment);
        size_t offset = 0;
        mRequests->_basePtr = allocate(r_size);
        for (auto &re : mRequests->_mem_requests) {
            auto sz = re._element_size * re._num_elements;

            if (re._ptr_out != nullptr) {
                auto cptr = mRequests->_basePtr.get() + offset;
                size_t cptr_avail_size = r_size - offset;
                if (re._type & REQUEST_BIND) {
                    cptr = reinterpret_cast<uint8_t*>(*reinterpret_cast<void **>(re._ptr_out));
                    cptr_avail_size = sz;
                } else {
                    *reinterpret_cast<void **>(re._ptr_out) = cptr;
                }
                std::cout << "ALLOCATED=" << static_cast<void*>(cptr) << ", size=" << re._element_size * re._num_elements << "\n";
                getQueue(REGION_AUTO)->iterate_binded(re, [](MemRequest & reference, MemRequest & binded) {
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
            offset += ALIGN(sz + re._padding, re._alignment);
        }
        return offset;
    }

    /**
     * @brief calculates size required for all requests, allocates memory and updates pointers
     */
    void commit() {
        // getTotalBytes();
        size_t heap_offset = 0;
        for (auto queue : _mem_queues) {
            if (queue.second->calcSize() != 0) {
                heap_offset = ALIGN(allocateRegion(queue.second), _page_alignment);
                std::cout << "heap_offset " << rRegionToStr(queue.first) << ": " << heap_offset << std::endl;
            }
        }
#ifdef GNA_HEAP_PROFILER
        memoryDump();
#endif
    }

 protected:
    std::shared_ptr<uint8_t> allocate(size_t bytes) {
        std::shared_ptr<uint8_t> sp(_allocator.allocate(bytes), [=](uint8_t *p) {
            _allocator.deallocate(p, bytes);
        });
        std::fill(sp.get(), sp.get() + bytes, 0);
        return sp;
    }

 protected:
    void expandBindRequests(std::shared_ptr<GNAMemRequestsQueue> mRequests) {
        // 1st stage -- looking for expandable bind requests:
        for (auto &originated : mRequests->_mem_requests) {
            if (originated._type & REQUEST_BIND) continue;
            size_t offset = 0;
            getQueue(REGION_AUTO)->iterate_binded(originated, [&](MemRequest & reference, MemRequest & binded) {
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
    }

#ifdef GNA_HEAP_PROFILER
    void memoryDump() {
        for (auto queue : _mem_queues) {
            std::ofstream dumpFile("gna_memory_requests_" + std::string(rRegionToStr(queue.first)) + ".txt", std::ios::out);
            for (auto &re : queue.second->_mem_requests) {
            dumpFile << "region: " << rRegionToStr(re._region) << ", "
                    << "type: " << std::setw(17) << rTypeToStr(re._type) << " "
                    << "ptr_in: " << std::setw(15) << re._ptr_in << " "
                    << "ptr_out: " << std::setw(15) << re._ptr_out << " "
                    << std::setw(8) << re._num_elements << ", "
                    << static_cast<int>(re._element_size) << ", "
                    << re._padding << ", "
                    << std::setw(3) << re._alignment << ", "
                    << std::setw(8) << re._offset << ", "
                    << std::endl;
            }
        }
    }
#endif
};

}  // namespace memory
}  // namespace GNAPluginNS
