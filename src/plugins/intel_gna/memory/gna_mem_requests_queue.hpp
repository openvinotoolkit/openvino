// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <vector>
#include <algorithm>
#include <functional>
#include <memory>

#include <ie_api.h>
#include <legacy/ie_layers.h>
#include "gna_plugin_log.hpp"
#include "gna_mem_requests.hpp"
#include "gna_lib_ver_selector.hpp"
#include "memory_solver.hpp"

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
    explicit GNAMemRequestsQueue(rRegion region) : _region_type(region) {
    }
    virtual ~GNAMemRequestsQueue() {}

    rRegion _region_type;
    size_t _size = 0;
    std::vector<MemRequest> _mem_requests;
    std::list<std::vector<char>> _local_storage;
    std::shared_ptr<uint8_t> _basePtr = nullptr;

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
    rRegion regionType() const {
        return _region_type;
    }

    std::vector<MemRequest> & futureHeap()  {
        return _mem_requests;
    }

    std::list<std::vector<char>> &localStorage() {
        return _local_storage;
    }

    virtual size_t calcSize(bool isCompact = false) {
        _size = 0;
        for (auto &re : _mem_requests) {
            if (re._type == REQUEST_BIND || re._ptr_out == nullptr) continue;
            _size += ALIGN(re._num_elements * re._element_size + re._padding, re._alignment);
        }
        return _size;
    }

    size_t getSize() {
        return _size;
    }

    void *getBasePtr() {
        return _basePtr.get();
    }

    template<class T>
    void iterate_binded(GNAPluginNS::memory::MemRequest & reference, const T & visitor) {
        for (auto &re : _mem_requests) {
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
};

class GNAMemRequestsInputsQueue : public GNAMemRequestsQueue {
public:
    explicit GNAMemRequestsInputsQueue() : GNAMemRequestsQueue(REGION_INPUTS) {
    }
};

class GNAMemRequestsOutputsQueue : public GNAMemRequestsQueue {
public:
    explicit GNAMemRequestsOutputsQueue() : GNAMemRequestsQueue(REGION_OUTPUTS) {
    }
};

class GNAMemRequestsScratchQueue : public GNAMemRequestsQueue {
public:
    explicit GNAMemRequestsScratchQueue() : GNAMemRequestsQueue(REGION_SCRATCH) {
    }
    /**
     * @brief optimize memory region by reusing buffers
     */
    size_t calcSize(bool isCompact = false) override {
        if (isCompact) {
            _size = 0;
            std::vector<MemorySolver::Box> boxes;
            for (size_t i = 0; i < _mem_requests.size(); ++i) {
                // skipping BIND, cross-region and empty requests
                if (_mem_requests[i]._type & REQUEST_BIND || _mem_requests[i]._ptr_out == nullptr) {
                    continue;
                }

                auto original_with_pad = ALIGN(_mem_requests[i]._num_elements * _mem_requests[i]._element_size + _mem_requests[i]._padding,
                                                _mem_requests[i]._alignment);
                int start = _mem_requests[i]._life_limits.first;
                int stop = _mem_requests[i]._life_limits.second;

                boxes.push_back({start, stop, static_cast<int64_t>(original_with_pad), static_cast<int64_t>(i)});
            }

            MemorySolver memSolver(boxes);
            _size = memSolver.solve();

            // setting offsets
            for (auto const & box : boxes) {
                _mem_requests[box.id]._offset = memSolver.getOffset(box.id);
            }
            return _size;
        } else {
            return GNAMemRequestsQueue::calcSize(isCompact);
        }
    }
};

class GNAMemRequestsReadOnlyQueue : public GNAMemRequestsQueue {
public:
    explicit GNAMemRequestsReadOnlyQueue() : GNAMemRequestsQueue(REGION_RO) {
    }
};

class GNAMemRequestsStatesQueue : public GNAMemRequestsQueue {
public:
    explicit GNAMemRequestsStatesQueue() : GNAMemRequestsQueue(REGION_STATES) {
    }
};

class GNAMemRequestsBindingsQueue : public GNAMemRequestsQueue {
public:
    explicit GNAMemRequestsBindingsQueue() : GNAMemRequestsQueue(REGION_AUTO) {
    }
};

}  // namespace memory
}  // namespace GNAPluginNS
