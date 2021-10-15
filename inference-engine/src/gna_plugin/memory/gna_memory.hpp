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
#include <iostream>
#include "gna_lib_ver_selector.hpp"
#include "memory_solver.hpp"
#include "backend/dnn_components.hpp"

#ifdef GNA_HEAP_PROFILER
#include <iomanip>
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
     * @brief calculates size required for all requests, allocates memory and updates pointers
     */
    void commit() {
        // 1st stage -- looking for expandable bind requests:
        auto setOffsets = [&](std::function<bool(MemRequest & request)> filter) {
            size_t region_offset = 0;
            for (auto &originated : _future_heap) {
                if (filter(originated) || originated._ptr_out == nullptr) continue;
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
                originated._offset = region_offset;
                region_offset += ALIGN(originated._num_elements * originated._element_size + originated._padding, originated._alignment);
            }
        };

        setOffsets([](GNAPluginNS::memory::MemRequest & request) {
            return (request._region != REGION_RW) || (request._type & REQUEST_BIND);
        });

        setOffsets([](GNAPluginNS::memory::MemRequest & request) {
            return (request._region != REGION_RO) || (request._type & REQUEST_BIND);
        });


        // allocation with memory setting to 0 internally
        heap = allocate(getTotalBytes());
        auto allocateSection = [&](std::function<bool(MemRequest & request)> filter, size_t section_offset) {
            for (auto &re : _future_heap) {
                if (re._type == REQUEST_BIND) continue;
                if (filter(re)) continue;

                auto sz = re._element_size * re._num_elements;

                size_t offset = section_offset + re._offset;
                if (re._ptr_out != nullptr) {
                    auto cptr = heap.get() + offset;
                    size_t cptr_avail_size = _total - offset;
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
            }
        };

        allocateSection([](GNAPluginNS::memory::MemRequest & request) {
            // TODO: consume bind requests separately from storage type
            return !(request._type & REQUEST_BIND) && (request._region != REGION_RW);
        }, 0);

        allocateSection([](GNAPluginNS::memory::MemRequest & request) {
            return (request._type & REQUEST_BIND) || request._region != REGION_RO;
        }, _rw_section_size);
    }

    void setExecutionOrder(GNAPluginNS::backend::DnnComponents &dnnComponents) {
        for (auto &re : _future_heap) {
            // default execution time from first to the last component
            std::pair<uint16_t, uint16_t> limits {0, dnnComponents.components.size()};

            // try to find the component with pointer
            auto dnn_comp = (re._ptr_out != nullptr) ? dnnComponents.findFirstComponentWithPtr(re._ptr_out) : nullptr;
            if (dnn_comp != nullptr) {
                limits = {dnn_comp->execOrder, dnn_comp->execOrder};
            }

            // looking for bind requests to identify the latest component using this buffer
            if (!(re._type & REQUEST_BIND)) {
                iterate_binded(re, [&](MemRequest & reference, MemRequest & binded) {
                    auto comp_binded = (binded._ptr_out != nullptr) ? dnnComponents.findFirstComponentWithPtr(binded._ptr_out) : nullptr;
                    uint16_t order_binded = (comp_binded != nullptr) ? comp_binded->execOrder : dnnComponents.components.size();
                    limits.second = std::max(limits.second, order_binded);
                });
            }
            re._life_limits = limits;
        }
        _is_compact_mode = true;
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
    size_t getSectionSizeOptimized(GNAPluginNS::memory::rRegion regType) {
        size_t memSize = 0;
        switch (regType) {
            case REGION_AUTO:
            case REGION_RW:
            case REGION_RO: {
                    std::vector<MemorySolver::Box> boxes;
                    for (int i = 0; i < _future_heap.size(); ++i) {
                        if (_future_heap[i]._type & REQUEST_BIND || _future_heap[i]._region != regType) {
                            continue;
                        }

                        auto original_with_pad = ALIGN(_future_heap[i]._num_elements * _future_heap[i]._element_size + _future_heap[i]._padding,
                                                       _future_heap[i]._alignment);
                        int start = _future_heap[i]._life_limits.first;
                        int stop = _future_heap[i]._life_limits.second;

                        boxes.push_back({start, stop, static_cast<int64_t>(original_with_pad), i});
                    }
                    MemorySolver memSolver(boxes);
                    memSize = memSolver.solve();

                    // setting offsets
                    for (auto box : boxes) {
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
#ifdef GNA_HEAP_PROFILER
        std::cout << "ro_section_size: " << _ro_section_size << std::endl;
        std::cout << "rw_section_size: " << _rw_section_size << std::endl;
        std::cout << "total: " << _total << std::endl;
#endif
        _rw_section_size = ALIGN(_rw_section_size, _page_alignment);
        _ro_section_size = ALIGN(_ro_section_size, _page_alignment);
        _total = _rw_section_size + _ro_section_size;
#ifdef GNA_HEAP_PROFILER
        std::cout << "Aligned ro_section_size: " << _ro_section_size << std::endl;
        std::cout << "Aligned rw_section_size: " << _rw_section_size << std::endl;
#endif
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
