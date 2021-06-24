// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>
#include <list>
#include <vector>

#include <vpu/utils/enums.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/middleend/allocator/structs.hpp>
#include <vpu/middleend/allocator/shaves.hpp>

namespace vpu {

//
// UsedMemory
//

struct UsedMemory final {
    int BSS = 0;
    int CMX = 0;
    int blob = 0;
    int input = 0;
    int output = 0;
};

void printTo(std::ostream& os, const UsedMemory& usedMemory);
void printTo(DotLabel& lbl, const UsedMemory& usedMemory);

//
// AllocationResult
//

VPU_DECLARE_ENUM(AllocationStatus,
    OK,
    SHAVES_FAILED,
    DATA_FAILED)

struct AllocationResult final {
    AllocationStatus status = AllocationStatus::OK;
    Stage failedStage = nullptr;
    Data failedData = nullptr;
};

//
// DeallocationMode
//

//
// The following deallocation modes are supported to speed-up performance:
//   * JustFree - Usual data deallocation scheme
//   * MoveFromCMX  - Simple check and reallocation to DDR if tensor does not meet CMX requirements
//

VPU_DECLARE_ENUM(DeallocationMode,
    JustFree,
    MoveFromCMX)

//
// Allocator
//

class Allocator final {
public:
    Allocator();

    void setBatchSize(int batchSize) { _modelBatchSize = batchSize; }

    void reset();

    /**
     * Allocates memory for single data node
     */
    bool allocateData(const Data& data);
    ShapeLocation allocateShape(const Data& data);
    void freeData(const Data& data, DeallocationMode mode = DeallocationMode::JustFree);

    void selfCheck();

    UsedMemory usedMemoryAmount() const;
    std::size_t freeMemoryAmount(const MemoryType& type) const;

    DataVector getAllocatedDatas(MemoryType memType) const;

    void setNeedToAllocNonIntermData() { _needToAllocNonIntermData = true; }
    /**
     * Allocates memory for the whole vector of data nodes
     */
    AllocationResult preprocess(const Model& model);

    DataSet& getCandidatesForCMX() { return _candidatesForCMX; }
    bool removeCMXCandidates(const Data& data);

    std::size_t freeCMXMemoryAmount() const;

    AllocatorForShaves& getAllocatorOfShaves() { return _allocatorOfShaves; }

private:
    allocator::MemChunk* allocateMem(MemoryType memType, int size, int inUse);
    void freeMem(allocator::MemChunk* chunk);

    allocator::MemChunk* addNewChunk(allocator::MemoryPool& pool, MemoryType memType, int offset, int pointer, int size, int inUse);
    allocator::MemChunk* checkMemPool(allocator::MemoryPool& pool, MemoryType memType, int size, int inUse);

    void extractDatas(MemoryType memType, const DataSet& from, DataVector& out) const;

    void updateChildDataAllocation(const Data& data);

private:
    int _modelBatchSize = 1;

    int _maxCmxSize = 0;

    allocator::MemoryPool _ddrMemoryPool;
    allocator::MemoryPool _cmxMemoryPool;
    EnumMap<MemoryType, allocator::MemoryPool*> _memPools;

    AllocatorForShaves _allocatorOfShaves;

    DataSet _allocatedData;
    DataSet _allocatedIntermData;

    DataMap<allocator::MemChunk*> _memChunksPerData;

    std::map<std::pair<DimVector, DimValues>, int> _staticShapeOffsets;

    int _blobMemOffset = 0;
    int _inputMemOffset = 0;
    int _outputMemOffset = 0;

    /**
     * Means that Model::_datas list was changed in some way
     */
    bool _needToAllocNonIntermData = true;

    DataSet _candidatesForCMX;
};

int calcAllocationSize(const Data& data);

}  // namespace vpu
