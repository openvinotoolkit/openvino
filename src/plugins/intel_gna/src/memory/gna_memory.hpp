// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_memcpy.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "gna_allocator.hpp"
#include "gna_lib_ver_selector.hpp"
#include "gna_mem_requests.hpp"
#include "gna_mem_requests_queue.hpp"
#include "log/log.hpp"
#include "memory/gna_allocator.hpp"
#include "memory_solver.hpp"

#ifdef GNA_MEMORY_DUMP
#    include <iomanip>
#endif

namespace ov {
namespace intel_gna {
namespace memory {

class GNAFloatAllocator : public std::allocator<uint8_t> {
public:
    void setTag(void*, memory::rRegion) {}
};

class GNAMemoryInterface {
public:
    virtual GNAMemRequestsQueue* getQueue(rRegion region) = 0;
    virtual GNAMemRequestsQueue* getQueue(void* ptr) = 0;
    virtual void commit(bool isCompact = false) = 0;
    virtual std::pair<bool, uint32_t> getOffsetForMerged(void* ptr) = 0;
    virtual size_t getRegionBytes(rRegion region) = 0;
    virtual size_t getDataMemAlignment() const = 0;
    virtual ~GNAMemoryInterface() = default;
};

/**
 * @brief encapsulate various request to allocate GNA specific memory,
 * in order to issue single allocation call and configure actual pointers in requests
 * @tparam Allocator - a GNAAllocator in case of actual HW offloads
 */
template <class Allocator = GNAAllocator>
class GNAMemory : public GNAMemoryInterface {
protected:
    std::map<rRegion, std::unique_ptr<GNAMemRequestsQueue>> _mem_queues;
    size_t _total = 0;
    Allocator _allocator;
    size_t _data_alignment;
    size_t _page_alignment;
    bool _is_compact_mode = false;

private:
    void initMemQueses() {
        _mem_queues[REGION_RO] = tools::make_unique<GNAMemRequestsReadOnlyQueue>(_data_alignment);
        _mem_queues[REGION_INPUTS] = tools::make_unique<GNAMemRequestsInputsQueue>(_data_alignment);
        _mem_queues[REGION_OUTPUTS] = tools::make_unique<GNAMemRequestsOutputsQueue>(_data_alignment);
        _mem_queues[REGION_SCRATCH] = tools::make_unique<GNAMemRequestsScratchQueue>(_data_alignment);
        _mem_queues[REGION_STATES] = tools::make_unique<GNAMemRequestsStatesQueue>(_data_alignment);
        _mem_queues[REGION_AUTO] = tools::make_unique<GNAMemRequestsBindingsQueue>(_data_alignment);
    }

public:
    explicit GNAMemory(size_t dataAlignment = 1, size_t pageAlignment = 1)
        : _data_alignment(dataAlignment),
          _page_alignment(pageAlignment) {
        initMemQueses();
    }

    explicit GNAMemory(const Allocator& a, size_t dataAlignment = 1, size_t pageAlignment = 1)
        : _allocator(a),
          _data_alignment(dataAlignment),
          _page_alignment(pageAlignment) {
        initMemQueses();
    }

    virtual ~GNAMemory() {
        // we have to deallocate regions before _allocator is destoyed
        _mem_queues.clear();
    }

    /**
     * @brief enables memory optimization (compact mode). This mode can be enable in plugin configuration (COMPACT_MODE
     * = Yes)
     */
    void setCompactMode(bool isCompact) {
        _is_compact_mode = isCompact;
    }

    /**
     * @brief calculates size required for all requests, allocates memory and updates pointers
     */
    void commit(bool isCompact = false) override {
        setCompactMode(isCompact);

        for (const auto& queue : _mem_queues) {
            // 1st stage -- looking for expandable bind requests:
            expandBindings(queue.second.get());

            // 2nd stage -- setup offsets:
            setRegionOffsets(queue.second.get());

            if (queue.second->calcSize(_is_compact_mode) != 0) {
                // 3rd stage -- allocation total memory setting to 0 internally
                queue.second->_basePtr = allocate(ALIGN(queue.second->getSize(), _page_alignment));
                log::debug() << rRegionToStr(queue.second->_region_type) << "("
                             << static_cast<void*>(queue.second->_basePtr.get()) << ")"
                             << " allocated: " << ALIGN(queue.second->getSize(), _page_alignment) << std::endl;
                // 4th stage -- setting proper GNA memory region tag for embedded TLV export
                _allocator.setTag(queue.second->getBasePtr(), queue.first);
                // 5th stage -- store data and updates pointers
                allocateRegion(queue.second.get());
            }
        }
#ifdef GNA_MEMORY_DUMP
        memoryDump();
#endif
    }

    GNAMemRequestsQueue* getQueue(rRegion region) override {
        return _mem_queues[region].get();
    }

    GNAMemRequestsQueue* getQueue(void* ptr) override {
        for (auto& queuePair : _mem_queues) {
            const auto offset = queuePair.second->getOffset(ptr);
            if (offset.first) {
                return queuePair.second.get();
            }
        }
        return nullptr;
    }

    std::pair<bool, uint32_t> getOffsetForMerged(void* ptr) override {
        uint32_t curOffset = 0;
        for (auto& queuePair : _mem_queues) {
            const auto offset = queuePair.second->getOffset(ptr);
            if (offset.first) {
                curOffset += offset.second;
                return {true, curOffset};
            }
            const auto size = queuePair.second->getSize();
            curOffset += ALIGN(static_cast<uint32_t>(size), static_cast<uint32_t>(_data_alignment));
        }
        return {false, 0};
    }

    size_t getRegionBytes(rRegion region) override {
        return ALIGN(getQueue(region)->calcSize(), _page_alignment);
    }

    template <class T>
    void iterate_binded(memory::MemRequest& reference, const T& visitor) {
        for (auto& re : getQueue(REGION_AUTO)->_mem_requests) {
            if ((re._type & REQUEST_BIND) && (re._ptr_in == reference._ptr_out)) {
                // log::trace() << "  [binded=" << rTypeToStr(re._type) << ", ptr=" << re._ptr_out <<"]\n";
                visitor(reference, re);
                // primitive loop check
                if (re._ptr_in == re._ptr_out)
                    continue;
                // TODO: no circular dependency checking, only tree-style dependency with loops supported
                iterate_binded(re, visitor);
            }
        }
#ifdef GNA_MEMORY_DUMP
        memoryDump();
#endif
    }

    size_t getDataMemAlignment() const override {
        return _data_alignment;
    }

protected:
    std::shared_ptr<uint8_t> allocate(size_t bytes) {
        Allocator nA = _allocator;
        std::shared_ptr<uint8_t> sp(_allocator.allocate(bytes), [nA, bytes](uint8_t* p) mutable {
            nA.deallocate(p, bytes);
        });
        std::fill(sp.get(), sp.get() + bytes, 0);
        return sp;
    }

    /**
     * @brief expand BIND and (BIND | ) requests. Align size(_padding), set execution order
     */
    void expandBindings(GNAMemRequestsQueue* mRequests) {
        for (auto& originated : mRequests->_mem_requests) {
            // skipping bind requests to avoid duplications
            if (originated._type & REQUEST_BIND)
                continue;

            size_t offset = 0;
            iterate_binded(originated, [&](MemRequest& reference, MemRequest& binded) {
                // aligning sizes
                if (&originated == &reference)
                    offset = 0;

                offset += binded._offset;
                auto current = offset + ALIGN(binded._num_elements * binded._element_size, binded._alignment);
                auto original_no_pad =
                    ALIGN(originated._num_elements * originated._element_size, originated._alignment);
                auto original_with_pad =
                    ALIGN(originated._num_elements * originated._element_size + originated._padding,
                          originated._alignment);

                originated._padding =
                    ALIGN(std::max(original_with_pad, current), originated._alignment) - original_no_pad;

                // set execution order
                originated._life_limits.first = std::min(originated._life_limits.first, binded._life_limits.first);
                originated._life_limits.second = std::max(originated._life_limits.second, binded._life_limits.second);
            });
        }
    }

    /**
     * @brief set offsets for specific region
     */
    size_t setRegionOffsets(GNAMemRequestsQueue* mRequests) {
        size_t region_offset = 0;
        for (auto& re : mRequests->_mem_requests) {
            if (re._type & REQUEST_BIND || re._ptr_out == nullptr)
                continue;
            re._offset = region_offset;
            region_offset += ALIGN(re._num_elements * re._element_size + re._padding, re._alignment);
        }
        return region_offset;
    }

    /**
     * @brief allocates memory and updates pointers
     */
    void allocateRegion(GNAMemRequestsQueue* mRequests) {
        size_t r_size = ALIGN(mRequests->getSize(), _page_alignment);
        for (auto& re : mRequests->_mem_requests) {
            // skipping Bind, crossregion and empty requests
            if (re._type == REQUEST_BIND || re._ptr_out == nullptr)
                continue;

            auto cptr = mRequests->_basePtr.get() + re._offset;
            size_t cptr_avail_size = r_size - re._offset;
            auto sz = re._element_size * re._num_elements;
            if (re._type & REQUEST_BIND) {
                cptr = reinterpret_cast<uint8_t*>(*reinterpret_cast<void**>(re._ptr_out));
                cptr_avail_size = sz;
            } else {
                *reinterpret_cast<void**>(re._ptr_out) = cptr;
            }
            iterate_binded(re, [](MemRequest& reference, MemRequest& binded) {
                *reinterpret_cast<void**>(binded._ptr_out) =
                    binded._offset + reinterpret_cast<uint8_t*>(*reinterpret_cast<void**>(reference._ptr_out));
                binded._num_elements = reference._num_elements;
                binded._element_size = reference._element_size;
            });

            log::debug() << static_cast<void*>(cptr) << "(" << sz + re._padding << ")" << std::endl;
            switch (re._type & ~REQUEST_BIND) {
            case REQUEST_ALLOCATE:
                break;
            case REQUEST_STORE: {
                if (re._ptr_in != nullptr) {
                    ie_memcpy(cptr, cptr_avail_size, re._ptr_in, sz);
                } else {
                    size_t of = 0;
                    for (size_t i = 0; i < re._num_elements; i++, of += re._element_size) {
                        std::copy(std::begin(re._data), std::end(re._data), cptr + of);
                    }
                }
                break;
            }
            case REQUEST_INITIALIZER: {
                re._initializer(cptr, sz);
                break;
            }
            }
        }
    }

#ifdef GNA_MEMORY_DUMP
    void memoryDump() {
        for (const auto& queue : _mem_queues) {
            std::ofstream dumpFile("gna_memory_requests_" + rRegionToStr(queue.first) + ".txt", std::ios::out);
            for (auto& re : queue.second->_mem_requests) {
                dumpFile << "region: " << rRegionToStr(re._region) << ", "
                         << "type: " << std::setw(17) << rTypeToStr(re._type) << " "
                         << "ptr_in: " << std::setw(15) << re._ptr_in << " "
                         << "ptr_out: " << std::setw(15) << re._ptr_out << " " << std::setw(8) << re._num_elements
                         << ", " << static_cast<int>(re._element_size) << ", " << re._padding << ", " << std::setw(3)
                         << re._alignment << ", " << std::setw(8) << re._offset << ", "
                         << "life_time: " << re._life_limits.first << ":" << re._life_limits.second << ", "
                         << std::endl;
            }
        }
    }
#endif
};

}  // namespace memory
}  // namespace intel_gna
}  // namespace ov
