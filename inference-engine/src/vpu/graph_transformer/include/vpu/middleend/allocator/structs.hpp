// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <vector>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/small_vector.hpp>
#include <vpu/model/data.hpp>

namespace vpu {


//
// Common allocation constants
//

const int CMX_SLICE_SIZE = 128 * 1024;
const int DATA_ALIGNMENT = 64;
const int CMX_SHAVE_BUFFER_SIZE = 100 * 1024;

//
// Allocator Structs
//

namespace allocator {

struct MemChunk final {
    MemoryType memType = MemoryType::DDR;
    int pointer = 0;
    int offset = 0;
    int size = 0;
    int inUse = 0;

    std::list<MemChunk>::iterator _posInList;
};

struct FreeMemory final {
    int offset = 0;
    int size = 0;
};

struct MemoryPool final {
    int curMemOffset = 0;
    int memUsed = 0;
    std::list<MemChunk> allocatedChunks;
    SmallVector<FreeMemory> freePool;

    void clear() {
        curMemOffset = 0;
        memUsed = 0;
        allocatedChunks.clear();
        freePool.clear();
    }
};

}  // namespace allocator


}  // namespace vpu
