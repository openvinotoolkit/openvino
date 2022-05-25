// Copyright (C) 2018-2022 Intel Corporation
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
#include <iostream>
#include "gna_lib_ver_selector.hpp"
#include "memory_solver.hpp"
#include "gna_plugin_log.hpp"
#include "memory/gna_allocator.hpp"

#ifdef GNA_HEAP_PROFILER
#include <iomanip>
#include <fstream>
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
protected:
    std::vector<MemRequest> _future_heap;
    std::list<std::vector<char>> _local_storage;
    size_t _total = 0;
    size_t _rw_section_size = 0;
    size_t _ro_section_size = 0;
    Allocator _allocator;
    std::shared_ptr<uint8_t> heap = nullptr;
    size_t _page_alignment = 1;
    bool _is_compact_mode = false;

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
     * @brief enables memory optimization (compact mode). This mode can be enable in plugin configuration (COMPACT_MODE = Yes)
     */
    void setCompactMode(bool isCompact) {
        _is_compact_mode = isCompact;
    }

    /**
     * @brief calculates size required for all requests, allocates memory and updates pointers
     */
    void commit(bool isCompact = false) {
        setCompactMode(isCompact);

        // 1st stage -- looking for expandable bind requests:
        expandBindings();

        // 2nd stage -- setup offsets:
        setRegionOffsets(REGION_RO);
        setRegionOffsets(REGION_RW);

        // 3rd stage -- allocation total memory setting to 0 internally
        heap = allocate(getTotalBytes());

        // 4th stage -- store data and updates pointers
        allocateRegion(REGION_RW, 0);
        allocateRegion(REGION_RO, _rw_section_size);
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
                // std::cout << "  [binded=" << rTypeToStr(re._type) << ", ptr=" << re._ptr_out <<"]\n";
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
    /**
     * @brief expand BIND and (BIND | ) requests. Align size(_padding), set execution order
     */
    void expandBindings() {
        for (auto &originated : _future_heap) {
            // skipping bind requests to avoid duplications
            if (originated._type & REQUEST_BIND) continue;

            size_t offset = 0;
            iterate_binded(originated, [&](MemRequest & reference, MemRequest & binded) {
                // aligning sizes
                if (&originated == &reference) offset = 0;

                offset += binded._offset;
                auto current = offset + ALIGN(binded._num_elements * binded._element_size, binded._alignment);
                auto original_no_pad = ALIGN(originated._num_elements * originated._element_size, originated._alignment);
                auto original_with_pad = ALIGN(originated._num_elements * originated._element_size + originated._padding, originated._alignment);

                originated._padding = ALIGN(std::max(original_with_pad, current), originated._alignment) - original_no_pad;

                // set execution order
                originated._life_limits.first = std::min(originated._life_limits.first, binded._life_limits.first);
                originated._life_limits.second = std::max(originated._life_limits.second, binded._life_limits.second);
            });
        }
    }

    /**
     * @brief set offsets for specific region
     */
    size_t setRegionOffsets(GNAPluginNS::memory::rRegion regType) {
        size_t region_offset = 0;
        for (auto &re : _future_heap) {
            if (re._region != regType || re._type & REQUEST_BIND || re._ptr_out == nullptr) continue;

            re._offset = region_offset;
            region_offset += ALIGN(re._num_elements * re._element_size + re._padding, re._alignment);
        }
        return region_offset;
    }

    /**
     * @brief allocates memory and updates pointers
     */
    void allocateRegion(GNAPluginNS::memory::rRegion regType, size_t baseOffset) {
        for (auto &re : _future_heap) {
            // skipping Bind, crossregion and empty requests
            if (re._region != regType || re._type == REQUEST_BIND || re._ptr_out == nullptr) continue;

            size_t offset = baseOffset + re._offset;
            auto cptr = heap.get() + offset;
            size_t cptr_avail_size = _total - offset;

            auto sz = re._element_size * re._num_elements;
            if (re._type & REQUEST_BIND) {
                cptr = reinterpret_cast<uint8_t*>(*reinterpret_cast<void **>(re._ptr_out));
                cptr_avail_size = sz;
            } else {
                *reinterpret_cast<void **>(re._ptr_out) = cptr;
            }
            iterate_binded(re, [](MemRequest & reference, MemRequest & binded) {
                *reinterpret_cast<void **>(binded._ptr_out) =
                    binded._offset + reinterpret_cast<uint8_t *>(*reinterpret_cast<void **>(reference._ptr_out));
                binded._num_elements = reference._num_elements;
                binded._element_size = reference._element_size;
            });

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
    }

    /**
     * @brief optimize memory region by reusing buffers
     */
    size_t getSectionSizeOptimized(GNAPluginNS::memory::rRegion regType) {
        size_t memSize = 0;
        switch (regType) {
            case REGION_AUTO:
            case REGION_RW:
            case REGION_RO: {
                    std::vector<MemorySolver::Box> boxes;
                    for (size_t i = 0; i < _future_heap.size(); ++i) {
                        // skipping BIND, cross-region and empty requests
                        if (_future_heap[i]._type & REQUEST_BIND || _future_heap[i]._region != regType || _future_heap[i]._ptr_out == nullptr) {
                            continue;
                        }

                        auto original_with_pad = ALIGN(_future_heap[i]._num_elements * _future_heap[i]._element_size + _future_heap[i]._padding,
                                                       _future_heap[i]._alignment);
                        int start = _future_heap[i]._life_limits.first;
                        int stop = _future_heap[i]._life_limits.second;

                        boxes.push_back({start, stop, static_cast<int64_t>(original_with_pad), static_cast<int64_t>(i)});
                    }
                    MemorySolver memSolver(boxes);
                    memSize = memSolver.solve();

                    // setting offsets
                    for (auto const & box : boxes) {
                        _future_heap[box.id]._offset = memSolver.getOffset(box.id);
                    }
                }
                break;

            default:
                break;
            }

        return memSize;
    }


#ifdef GNA_HEAP_PROFILER
    void memoryDump(std::function<bool(MemRequest & re)> filter) {
        std::ofstream dumpFile("gna_memory_requests.txt", std::ios::out);

        for (auto &re : _future_heap) {
            if (filter(re)) continue;
            dumpFile << ": " << " region: " << rRegionToStr(re._region) << ", "
                    << "type: " << std::setw(17) << rTypeToStr(re._type) << " "
                    << "ptr_in: " << std::setw(15) << re._ptr_in << " "
                    << "ptr_out: " << std::setw(15) << re._ptr_out << " "
                    << std::setw(8) << re._num_elements << ", "
                    << static_cast<int>(re._element_size) << ", "
                    << re._padding << ", "
                    << std::setw(3) << re._alignment << ", "
                    << std::setw(8) << re._offset << ", "
                    << "life_time: " << re._life_limits.first << ":" << re._life_limits.second << ", "
                    << std::endl;
        }
    }
#endif

    void updateSectionsSizes() {
        // count total size and size of read/write regions
        _rw_section_size = 0;
        _ro_section_size = 0;
#ifdef GNA_HEAP_PROFILER
        memoryDump([](GNAPluginNS::memory::MemRequest & request) {
            return false;
            });
#endif
        for (auto &re : _future_heap) {
            if (re._type & REQUEST_BIND || re._ptr_out == nullptr) continue;

            size_t current = ALIGN(re._num_elements * re._element_size + re._padding, re._alignment);
            if (re._region == REGION_RW) {
                _rw_section_size += current;
            } else {
                _ro_section_size += current;
            }
        }

        if (_is_compact_mode) {
            _rw_section_size = getSectionSizeOptimized(REGION_RW);
        }

        gnalog() << "ro_section_size: " << _ro_section_size << std::endl;
        gnalog() << "rw_section_size: " << _rw_section_size << std::endl;
        gnalog() << "total: " << _total << std::endl;

        _rw_section_size = ALIGN(_rw_section_size, _page_alignment);
        _ro_section_size = ALIGN(_ro_section_size, _page_alignment);
        _total = _rw_section_size + _ro_section_size;

        gnalog() << "Aligned ro_section_size: " << _ro_section_size << std::endl;
        gnalog() << "Aligned rw_section_size: " << _rw_section_size << std::endl;
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
