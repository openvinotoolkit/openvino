// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/allocator/allocator.hpp>

#include <unordered_set>
#include <algorithm>
#include <limits>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

//
// UsedMemory
//

void printTo(std::ostream& os, const UsedMemory& usedMemory) {
    os << "[" << std::endl;

    os << "BSS=" << usedMemory.BSS << std::endl;
    os << "CMX=" << usedMemory.CMX << std::endl;
    os << "blob=" << usedMemory.blob << std::endl;
    os << "input=" << usedMemory.input << std::endl;
    os << "output=" << usedMemory.output << std::endl;

    os << "]";
}

void printTo(DotLabel& lbl, const UsedMemory& usedMemory) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("BSS", usedMemory.BSS);
    subLbl.appendPair("CMX", usedMemory.CMX);
    subLbl.appendPair("blob", usedMemory.blob);
    subLbl.appendPair("input", usedMemory.input);
    subLbl.appendPair("output", usedMemory.output);
}

//
// Allocator
//

int calcAllocationSize(const Data& data) {
    return alignVal(data->totalByteSize(), DATA_ALIGNMENT);
}

Allocator::Allocator(): _allocatorOfShaves(_cmxMemoryPool) {
    const auto& env = CompileEnv::get();

    _maxCmxSize = env.resources.numCMXSlices * CMX_SLICE_SIZE;

    _memPools.emplace(MemoryType::DDR, &_ddrMemoryPool);
    _memPools.emplace(MemoryType::CMX, &_cmxMemoryPool);
}

namespace {

void updateChildDataAllocation(const Data& data, int offsetLimitation) {
    for (const auto& edge : data->childDataEdges()) {
        auto parent = edge->parent();
        auto child = edge->child();

        auto memoryOffset = parent->memoryOffset();

        if (edge->mode() == SharedDataMode::ROI) {
            auto parentStrides = parent->strides();
            const auto& offset = edge->attrs().get<DimValues>("offset");

            int byteOffset = 0;
            for (const auto& p : offset) {
                byteOffset += p.second * parentStrides[p.first];
            }

            memoryOffset += byteOffset;

            IE_ASSERT(memoryOffset + child->lastElemOffset() <= offsetLimitation);
        } else if (edge->mode() == SharedDataMode::Reshape) {
            IE_ASSERT(parent->checkStrides(StridesRequirement::compact()));
            IE_ASSERT(child->checkStrides(StridesRequirement::compact()));
        } else {
            IE_ASSERT(false) << "Unsupported enum value";
        }

        child->setAllocationInfo(parent->location(), memoryOffset);

        updateChildDataAllocation(child, offsetLimitation);
    }
}

}  // namespace

bool Allocator::allocateData(const Data& data) {
    //
    // Get location requirements
    //

    auto memoryType = data->memReqs();

    //
    // Fake data: make sure no memory is allocated
    //

    if (data->usage() == DataUsage::Fake) {
        if (_allocatedData.count(data) == 0) {
            IE_ASSERT(data->parentDataEdge() == nullptr);

            updateChildDataAllocation(data, 0);

            _allocatedData.emplace(data);
        }

        return true;
    }

    //
    // Input data
    //

    if (data->usage() == DataUsage::Input) {
        if (_allocatedData.count(data) == 0) {
            IE_ASSERT(data->parentDataEdge() == nullptr);

            auto finalByteSize = data->totalByteSize() * _modelBatchSize;

            data->setIOInfo(DataLocation::Input, alignVal(_inputMemOffset, DATA_ALIGNMENT));
            _inputMemOffset = alignVal(_inputMemOffset, DATA_ALIGNMENT) + finalByteSize;

            updateChildDataAllocation(data, DDR_MAX_SIZE);

            _allocatedData.emplace(data);
        }

        return memoryType == MemoryType::DDR;
    }

    //
    // Output data
    //

    if (data->usage() == DataUsage::Output) {
        if (_allocatedData.count(data) == 0) {
            IE_ASSERT(data->parentDataEdge() == nullptr);

            int finalByteSize = 0;
            if (data->attrs().getOrDefault<bool>("unbatched", false)) {
                finalByteSize = data->totalByteSize();
            } else {
                finalByteSize = data->totalByteSize() * _modelBatchSize;
            }

            data->setIOInfo(DataLocation::Output, alignVal(_outputMemOffset, DATA_ALIGNMENT));
            _outputMemOffset = alignVal(_outputMemOffset, DATA_ALIGNMENT) + finalByteSize;

            updateChildDataAllocation(data, DDR_MAX_SIZE);

            _allocatedData.emplace(data);
        }

        return memoryType == MemoryType::DDR;
    }

    //
    // Const data
    //

    if (data->usage() == DataUsage::Const) {
        if (_allocatedData.count(data) == 0) {
            IE_ASSERT(data->parentDataEdge() == nullptr);
            IE_ASSERT(data->checkStrides(StridesRequirement::compact()));
            IE_ASSERT(data->content() != nullptr);

            auto finalByteSize = calcAllocationSize(data);

            data->setAllocationInfo(DataLocation::Blob, _blobMemOffset);
            _blobMemOffset += finalByteSize;

            updateChildDataAllocation(data, DDR_MAX_SIZE);

            _allocatedData.emplace(data);
        }

        return memoryType == MemoryType::DDR;
    }

    //
    // Intermediate data must have producer and consumer(s)
    //

    if (data->usage() == DataUsage::Intermediate) {
        IE_ASSERT(data->producerEdge() != nullptr);
        IE_ASSERT(data->numConsumers() > 0);
    }

    //
    // Allocate parent data if any
    //

    if (auto parentEdge = data->parentDataEdge()) {
        auto parent = parentEdge->parent();

        auto parentMemType = parent->memReqs();
        IE_ASSERT(parentMemType == memoryType);

        // Parent will update all children.
        return allocateData(parent);
    }

    IE_ASSERT(data->parentDataEdge() == nullptr);

    //
    // Check if the data is already allocated
    //

    if (_allocatedIntermData.count(data) != 0) {
        auto it = _memChunksPerData.find(data);
        IE_ASSERT(it != _memChunksPerData.end());

        auto chunk = it->second;
        IE_ASSERT(chunk != nullptr);

        return chunk->memType == memoryType;
    }

    //
    // Calculate final buffer size
    //

    auto finalByteSize = calcAllocationSize(data);

    //
    // Allocate buffer in requested location
    //

    int inUse = 0;
    if (data->usage() == DataUsage::Temp) {
        inUse = 1;
    } else {
        loopOverData(data, [&inUse](const Data& subData) {
            inUse += subData->numConsumers();
            return DataLoopStatus::NextChild;
        });
    }
    IE_ASSERT(inUse >= 1);

    auto chunk = allocateMem(memoryType, finalByteSize, inUse);

    if (chunk == nullptr) {
        return false;
    }

    //
    // Update data allocation info
    //

    data->setAllocationInfo(chunk->memType == MemoryType::CMX ? DataLocation::CMX : DataLocation::BSS, chunk->pointer);

    auto offsetLimitation = (data->location() == DataLocation::CMX) ? _maxCmxSize : DDR_MAX_SIZE;
    updateChildDataAllocation(data, offsetLimitation);

    _memChunksPerData.emplace(data, chunk);
    _allocatedIntermData.emplace(data);

    return chunk->memType == memoryType;
}

void Allocator::freeData(const Data& data, DeallocationMode mode) {
    //
    // Release the chunk
    //

    auto topParent = data->getTopParentData();

    if (topParent->usage() == DataUsage::Intermediate ||
        topParent->usage() == DataUsage::Temp) {
        IE_ASSERT(_allocatedIntermData.count(topParent) > 0);

        auto it = _memChunksPerData.find(topParent);
        IE_ASSERT(it != _memChunksPerData.end());

        auto chunk = it->second;
        IE_ASSERT(chunk != nullptr);
        IE_ASSERT(chunk->inUse > 0);

        switch (mode) {
        case DeallocationMode::JustFree: {
            --chunk->inUse;

            if (chunk->inUse == 0) {
                freeMem(chunk);

                _memChunksPerData.erase(topParent);
                _allocatedIntermData.erase(topParent);
            }

            break;
        }

        case DeallocationMode::MoveFromCMX: {
            IE_ASSERT(chunk->memType == MemoryType::CMX);

            auto curChunkSz = chunk->size;
            auto inUse = chunk->inUse;

            freeMem(chunk);

            auto ddrChunk = allocateMem(MemoryType::DDR, curChunkSz, inUse);
            IE_ASSERT(ddrChunk!= nullptr);

            _memChunksPerData[data] = ddrChunk;

            data->setAllocationInfo(DataLocation::BSS, ddrChunk->pointer);
            updateChildDataAllocation(data, DDR_MAX_SIZE);

            break;
        }

        default:
            VPU_THROW_EXCEPTION << "Unsupported mode : " << mode;
        }
    }
}

void Allocator::selfCheck() {
    _allocatorOfShaves.selfCheck();

    for (const auto& p : _memPools) {
        if (!p.second->freePool.empty() || p.second->curMemOffset > 0) {
            VPU_THROW_EXCEPTION << "Internal error in " << p.first << " allocation";
        }
    }
}

UsedMemory Allocator::usedMemoryAmount() const {
    UsedMemory stats;

    stats.BSS = _ddrMemoryPool.memUsed;
    stats.CMX = _cmxMemoryPool.memUsed;
    stats.blob = _blobMemOffset;
    stats.input = _inputMemOffset;
    stats.output = _outputMemOffset;

    return stats;
}

std::size_t Allocator::freeDDRMemoryAmount() const {
    const auto& pool = _memPools.at(MemoryType::DDR);
    const auto offset = pool->curMemOffset;
    VPU_THROW_UNLESS(offset <= DDR_MAX_SIZE, "Out of bound offset for next free data in DDR: size = {}, while offset = {}", DDR_MAX_SIZE, offset);

    return DDR_MAX_SIZE - offset;
}

std::size_t Allocator::freeCMXMemoryAmount() const {
    const auto& pool = _memPools.at(MemoryType::CMX);
    const auto shavesCMX = _allocatorOfShaves.getLockedSHAVEs() * CMX_SLICE_SIZE;
    const auto offset = pool->curMemOffset + shavesCMX;
    VPU_THROW_UNLESS(offset <= _maxCmxSize, "Out of bound offset for next free data in CMX: size = {}, while offset = {}", _maxCmxSize, offset);

    return _maxCmxSize - offset;
}

std::size_t Allocator::freeMemoryAmount(const MemoryType& type) const {
    return type == MemoryType::CMX ? freeCMXMemoryAmount() : freeDDRMemoryAmount();
}

void Allocator::extractDatas(MemoryType memType, const DataSet& from, DataVector& out) const {
    for (const auto& data : from) {
        if (data->usage() != DataUsage::Intermediate)
            continue;

        auto it = _memChunksPerData.find(data);
        IE_ASSERT(it != _memChunksPerData.end());

        auto chunk = it->second;
        IE_ASSERT(chunk != nullptr);
        IE_ASSERT(chunk->inUse > 0);

        if (chunk->memType == memType) {
            out.emplace_back(data);
        }
    }
}

DataVector Allocator::getAllocatedDatas(MemoryType memType) const {
    DataVector out;

    if (memType == MemoryType::CMX) {
        out.reserve(_allocatedIntermData.size());
        extractDatas(memType, _allocatedIntermData, out);
    } else {
        out.reserve(_allocatedData.size() + _allocatedIntermData.size());
        extractDatas(memType, _allocatedData, out);
        extractDatas(memType, _allocatedIntermData, out);
    }

    return out;
}

allocator::MemChunk* Allocator::allocateMem(MemoryType memType, int size, int inUse) {
    VPU_THROW_UNLESS(size >= 0, "{} bytes to allocate have been requested, but only non-negative amount is supported", size);
    if (size == 0) {
        return nullptr;
    }

    auto& memPool =  _memPools.at(memType);

    //
    // Try to reuse already allocated memory
    //

    if (auto chunk = checkMemPool(*memPool, memType, size, inUse)) {
        memPool->memUsed = std::max(memPool->memUsed, chunk->offset + chunk->size);
        return chunk;
    }

    //
    // Check free space
    //

    const auto freeSpace = freeMemoryAmount(memType);
    if (static_cast<std::size_t>(size) > freeSpace) {
        return nullptr;
    }

    //
    // Allocate new chunk
    //

    int pointer = 0;
    if (memType == MemoryType::CMX) {
        IE_ASSERT(memPool->curMemOffset + size <= _maxCmxSize);
        pointer = _maxCmxSize - (memPool->curMemOffset + size);
    } else {
        pointer = memPool->curMemOffset;
    }

    auto chunk = addNewChunk(*memPool, memType, memPool->curMemOffset, pointer, size, inUse);
    IE_ASSERT(chunk != nullptr);

    memPool->curMemOffset += size;

    memPool->memUsed = std::max(memPool->memUsed, chunk->offset + chunk->size);

    return chunk;
}

void Allocator::freeMem(allocator::MemChunk* chunk) {
    IE_ASSERT(chunk != nullptr);

    auto& memPool =  _memPools.at(chunk->memType);

    allocator::FreeMemory newMem;
    newMem.offset = chunk->offset;
    newMem.size = chunk->size;

    while (true) {
        bool found = false;

        for (auto memPoolIt = memPool->freePool.begin(); memPoolIt != memPool->freePool.end(); ++memPoolIt) {
            IE_ASSERT(newMem.offset != memPoolIt->offset);

            if (newMem.offset + newMem.size == memPoolIt->offset) {
                //
                // [newMem][*memPoolIt] case
                // extend newMem to and remove memPoolIt
                //

                newMem.size += memPoolIt->size;

                memPool->freePool.erase(memPoolIt);

                found = true;
                break;
            } else if (memPoolIt->offset + memPoolIt->size == newMem.offset) {
                //
                // [*memPoolIt][newMem] case
                // extend newMem to and remove memPoolIt
                //

                newMem.offset = memPoolIt->offset;
                newMem.size += memPoolIt->size;

                memPool->freePool.erase(memPoolIt);

                found = true;
                break;
            }
        }

        if (!found) {
            if (newMem.offset + newMem.size == memPool->curMemOffset) {
                memPool->curMemOffset = newMem.offset;
            } else {
                memPool->freePool.emplace_back(newMem);
            }

            break;
        }
    }

    IE_ASSERT(chunk->_posInList != memPool->allocatedChunks.end());
    memPool->allocatedChunks.erase(chunk->_posInList);
}

allocator::MemChunk* Allocator::addNewChunk(allocator::MemoryPool& memPool, MemoryType memType, int offset, int pointer, int size, int inUse) {
    allocator::MemChunk newChunkValues;
    newChunkValues.memType = memType;
    newChunkValues.pointer = pointer;
    newChunkValues.offset = offset;
    newChunkValues.size = size;
    newChunkValues.inUse = inUse;
    auto it = memPool.allocatedChunks.emplace(memPool.allocatedChunks.end(), newChunkValues);

    auto newChunk = &memPool.allocatedChunks.back();
    newChunk->_posInList = it;

    return newChunk;
}

allocator::MemChunk* Allocator::checkMemPool(allocator::MemoryPool& memPool, MemoryType memType, int size, int inUse) {
    auto minMemSizeToUse = std::numeric_limits<size_t>::max();
    auto minMemIt = memPool.freePool.end();

    for (auto memPoolIt = memPool.freePool.begin(); memPoolIt != memPool.freePool.end(); ++memPoolIt) {
        if (memPoolIt->size >= size) {
            if (memPoolIt->size < minMemSizeToUse) {
                minMemSizeToUse = memPoolIt->size;
                minMemIt = memPoolIt;
            }
        }
    }

    if (minMemIt == memPool.freePool.end()) {
        return nullptr;
    }

    auto offset = minMemIt->offset + minMemIt->size - size;

    int pointer = 0;
    if (memType == MemoryType::DDR) {
        pointer = offset;
    } else {
        IE_ASSERT(offset + size <= _maxCmxSize);
        pointer = _maxCmxSize - offset - size;
    }

    auto chunk = addNewChunk(memPool, memType, offset, pointer, size, inUse);

    minMemIt->size -= size;

    if (minMemIt->size == 0) {
        memPool.freePool.erase(minMemIt);
    }

    return chunk;
}

void Allocator::reset() {
    const auto& env = CompileEnv::get();

    _maxCmxSize = env.resources.numCMXSlices * CMX_SLICE_SIZE;
    _allocatorOfShaves.reset();

    for (auto& pool : _memPools) {
        pool.second->clear();
    }

    _allocatedIntermData.clear();

    _memChunksPerData.clear();
}

AllocationResult Allocator::preprocess(const Model& model) {
    reset();

    if (_needToAllocNonIntermData) {
        _allocatedData.clear();
        _allocatedData.reserve(model->numDatas());

        _blobMemOffset = 0;
        _inputMemOffset = 0;
        _outputMemOffset = 0;

        for (const auto& data : model->datas()) {
            data->clearAllocation();
        }

        for (const auto& data : model->datas()) {
            if (data->usage() != DataUsage::Intermediate &&
                data->usage() != DataUsage::Temp) {
                if (!allocateData(data)) {
                    AllocationResult result;
                    result.status = AllocationStatus::DATA_FAILED;
                    result.failedStage = data->producer();
                    return result;
                }
            }
        }
    }

    _needToAllocNonIntermData = false;

    return AllocationResult();
}

bool Allocator::removeCMXCandidates(const vpu::Data& data) {
    auto it = _candidatesForCMX.find(data);

    if (it != _candidatesForCMX.end()) {
        IE_ASSERT(data->parentDataEdge() == nullptr);

        if (_allocatedIntermData.count(data) != 0) {
            if (auto producerEdge = data->producerEdge()) {
                if (producerEdge->portInd() == 0 &&
                    producerEdge->producer()->category() == StageCategory::HW) {
                    return true;
                }
            }

            freeData(data, DeallocationMode::MoveFromCMX);
        }

        loopOverData(data, [](const Data& subData) {
            subData->setMemReqs(MemoryType::DDR);
            return DataLoopStatus::NextChild;
        });

        _candidatesForCMX.erase(it);

        return true;
    } else {
        auto cmxDatas = getAllocatedDatas(MemoryType::CMX);

        for (const auto& cmxData : cmxDatas) {
            IE_ASSERT(cmxData->parentDataEdge() == nullptr);

            it = _candidatesForCMX.find(cmxData);

            if (it != _candidatesForCMX.end()) {
                freeData(cmxData, DeallocationMode::MoveFromCMX);

                loopOverData(cmxData, [](const Data& subData) {
                    subData->setMemReqs(MemoryType::DDR);
                    return DataLoopStatus::NextChild;
                });

                _candidatesForCMX.erase(it);

                // TODO: remove the first CMX candidate or remove all CMX candidates?
                return true;
            }
        }
    }

    return false;
}

}  // namespace vpu
