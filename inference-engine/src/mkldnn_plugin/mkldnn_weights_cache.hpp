// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_memory.h>

#include <unordered_map>
#include <functional>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <map>

// TODO: While CPU plugin has no ease way to clone graph object we use weight
//       caching in global Engine context to avoid tensor memory duplication.
//       For same cases it may be switched of (like for single stream execution)
//       When MKLDNNGraph clone function will be ready you may removed this
//       classes at all.

namespace MKLDNNPlugin {

class SimpleDataHash {
public:
    SimpleDataHash() {
        for (int i = 0; i < kTableSize; i++) {
            uint64_t c = i;
            for (int j = 0; j < 8; j++)
                c = ((c & 1) ? 0xc96c5795d7870f42 : 0) ^ (c >> 1);
            table[i] = c;
        }
    }
    // Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
    uint64_t hash(const unsigned char* data, size_t size) const {
        uint64_t crc = 0;
        for (size_t idx = 0; idx < size; idx++)
            crc = table[(unsigned char)crc ^ data[idx]] ^ (crc >> 8);

        return ~crc;
    }

protected:
    static const int kTableSize = 256;
    uint64_t table[kTableSize];
};

/**
 * Caching store of MKLDNNMemory objects
 * Will return a cached object or create new one
 *
 * Is a thread safe
 */
class MKLDNNWeightsSharing {
    struct MKLDNNMemoryInfo {
        typedef std::shared_ptr<MKLDNNMemoryInfo> Ptr;

        MKLDNNMemoryInfo(MKLDNNMemoryPtr memoryPtr, bool valid)
            : sharedMemory(memoryPtr)
            , valid(valid)
        {}

        std::mutex guard;
        std::weak_ptr<MKLDNNMemory> sharedMemory;
        std::atomic<bool> valid;
    };

public:
    typedef std::shared_ptr<MKLDNNWeightsSharing> Ptr;

    class MKLDNNSharedMemory {
    public:
        typedef std::shared_ptr<MKLDNNSharedMemory> Ptr;

        MKLDNNSharedMemory(std::unique_lock<std::mutex> && lock,
                           const MKLDNNMemoryInfo::Ptr & memory,
                           MKLDNNMemoryPtr newPtr = nullptr);

        operator MKLDNNMemoryPtr() const;
        bool isValid() const;
        void valid(bool b);

    private:
        std::unique_lock<std::mutex> lock;
        MKLDNNMemoryInfo::Ptr memory;
        MKLDNNMemoryPtr newPtr;
    };

    MKLDNNSharedMemory::Ptr findOrCreate(const std::string& key,
                                         std::function<MKLDNNMemoryPtr(void)> create,
                                         bool valid = true);

    MKLDNNSharedMemory::Ptr get(const std::string& key) const;

    static const SimpleDataHash& GetHashFunc () { return simpleCRC; }

protected:
    mutable std::mutex guard;
    std::unordered_map<std::string, MKLDNNMemoryInfo::Ptr> sharedWeights;
    static const SimpleDataHash simpleCRC;
};

/**
 * Collection of memory caching store per NUMA node(former socket)
 *
 * Is a thread safe
 */
class NumaNodesWeights {
public:
    NumaNodesWeights();

    MKLDNNWeightsSharing::Ptr& operator[](int i);
    const MKLDNNWeightsSharing::Ptr& operator[](int i) const;

private:
    std::map<int, MKLDNNWeightsSharing::Ptr> _cache_map;
};

}  // namespace MKLDNNPlugin
