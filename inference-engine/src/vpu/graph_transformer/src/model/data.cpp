// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data.hpp>

#include <array>
#include <algorithm>
#include <queue>
#include <memory>
#include <vector>
#include <unordered_map>
#include <map>
#include <string>
#include <set>

#include <precision_utils.h>
#include <ie_parallel.hpp>

#include <vpu/model/edges.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/backend/backend.hpp>

namespace vpu {

//
// DataContent
//

const void* CalculatedDataContent::getRaw() const {
    if (_temp.empty()) {
        _temp.resize(getTempBufSize(_baseContents));
        fillTempBuf(_baseContents, _temp.data());
        _baseContents.clear();
    }
    return _temp.data();
}

size_t CalculatedDataContent::getTempBufSize(const SmallVector<DataContent::Ptr, 2>&) const {
    return _desc.totalDimSize() * _desc.elemSize();
}

namespace {

class IeBlobContent final : public DataContent {
public:
    IeBlobContent(const ie::Blob::Ptr& blob, int repeat) : _blob(blob), _repeat(repeat) {}

protected:
    const void* getRaw() const override {
        IE_ASSERT(_desc.type() == DataType::FP16);

        if (_blobFp16 == nullptr) {
            _blobFp16 = getBlobFP16(_blob);
            _blob.reset();
        }

        if (_repeat == 1) {
            return _blobFp16->cbuffer();
        } else {
            if (_temp.empty()) {
                VPU_PROFILE(IeBlobContent);

                IE_ASSERT(_desc.totalDimSize() % _repeat == 0);

                auto origNumElems = _desc.totalDimSize() / _repeat;
                IE_ASSERT(origNumElems <= _blobFp16->size());

                auto origPtr = _blobFp16->cbuffer().as<const fp16_t*>();
                IE_ASSERT(origPtr != nullptr);

                _temp.resize(_desc.totalDimSize());

                ie::parallel_for(_repeat, [this, origPtr, origNumElems](int i) {
                    std::copy_n(origPtr, origNumElems, _temp.data() + i * origNumElems);
                });
            }

            return _temp.data();
        }
    }

private:
    mutable ie::Blob::Ptr _blob;
    int _repeat = 0;

    mutable ie::Blob::Ptr _blobFp16;
    mutable std::vector<fp16_t> _temp;
};

}  // namespace

DataContent::Ptr ieBlobContent(const ie::Blob::Ptr& blob, int repeat) {
    return std::make_shared<IeBlobContent>(blob, repeat);
}

namespace {

class ReplicatedContent final : public CalculatedDataContent {
public:
    ReplicatedContent(float val, int count) : _val(val), _count(count) {}

    ReplicatedContent(const DataContent::Ptr& origContent, int count) :
            CalculatedDataContent({origContent}), _count(count) {
    }

protected:
    size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>& baseContents) const override {
        if (baseContents.empty()) {
            return _count * sizeof(fp16_t);
        } else {
            IE_ASSERT(baseContents.size() == 1);
            IE_ASSERT(_desc.totalDimSize() % _count == 0);

            return _desc.totalDimSize() * sizeof(fp16_t);
        }
    }

    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(ReplicatedContent);

        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        if (baseContents.empty()) {
            std::fill_n(dstPtr, _count, ie::PrecisionUtils::f32tof16(_val));
        } else {
            IE_ASSERT(baseContents.size() == 1);
            IE_ASSERT(_desc.totalDimSize() % _count == 0);

            auto origCount = _desc.totalDimSize() / _count;
            auto origPtr = baseContents[0]->get<fp16_t>();
            IE_ASSERT(origPtr != nullptr);

            ie::parallel_for(_count, [origPtr, origCount, dstPtr](int i) {
                std::copy_n(origPtr, origCount, dstPtr + i * origCount);
            });
        }
    }

private:
    float _val = 0.0f;
    int _count = 0;
};

}  // namespace

DataContent::Ptr replicateContent(
        float val,
        int count) {
    return std::make_shared<ReplicatedContent>(val, count);
}

DataContent::Ptr replicateContent(
        const DataContent::Ptr& origContent,
        int count) {
    return std::make_shared<ReplicatedContent>(origContent, count);
}

namespace {

class ScaledContent final : public CalculatedDataContent {
public:
    ScaledContent(
            const DataContent::Ptr& origContent,
            float scale) :
            CalculatedDataContent({origContent}), _scale(scale) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(ScaledContent);

        IE_ASSERT(baseContents.size() == 1);

        auto totalSize = _desc.totalDimSize();

        auto origDesc = baseContents[0]->desc();
        IE_ASSERT(origDesc.type() == DataType::FP16);
        IE_ASSERT(origDesc.totalDimSize() == totalSize);

        auto srcPtr = baseContents[0]->get<fp16_t>();
        IE_ASSERT(srcPtr != nullptr);

        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        ie::parallel_for(totalSize, [this, srcPtr, dstPtr](int i) {
            dstPtr[i] = ie::PrecisionUtils::f32tof16(ie::PrecisionUtils::f16tof32(srcPtr[i]) * _scale);
        });
    }

private:
    float _scale = 0.0f;
};

}  // namespace

DataContent::Ptr scaleContent(
        const DataContent::Ptr& origContent,
        float scale) {
    return std::make_shared<ScaledContent>(origContent, scale);
}

//
// DataNode
//

Data DataNode::getTopParentData() const {
    auto topParent = handle_from_this();
    while (auto nextParent = topParent->parentData()) {
        topParent = nextParent;
    }
    return topParent;
}

DimValues DataNode::strides() const {
    if (_parentDataEdge != nullptr) {
        if (_parentDataEdge->mode() == SharedDataMode::ROI) {
            return _parentDataEdge->parent()->strides();
        }
    }

    return calcStrides(_desc, _requiredStrides);
}

int DataNode::totalByteSize() const {
    // IT doesn't have sence for child Data.
    IE_ASSERT(_parentDataEdge == nullptr);

    return calcTotalByteSize(_desc, strides());
}

int DataNode::elemOffset(const DimValues& coord) const {
    auto strides = this->strides();

    int res = 0;
    for (const auto& p : coord) {
        IE_ASSERT(_desc.dimsOrder().hasDim(p.first));
        IE_ASSERT(p.second < _desc.dim(p.first));
        res += p.second * strides[p.first];
    }

    return res;
}

int DataNode::lastElemOffset() const {
    DimValues lastElem;
    for (const auto& p : _desc.dims()) {
        lastElem.set(p.first, p.second - 1);
    }
    return elemOffset(lastElem);
}

bool DataNode::checkStrides(const StridesRequirement& reqs) const {
    return vpu::checkStrides(_desc, strides(), reqs);
}

void DataNode::updateRequiredStrides(const StridesRequirement& newReqs) {
    // There shouldn't be any Data<->Data edges.
    IE_ASSERT(_parentDataEdge == nullptr);
    IE_ASSERT(_childDataEdges.empty());

    auto prevReqs = _requiredStrides;

    StridesRequirement mergedReqs;
    for (int i = 0; i < _desc.numDims(); ++i) {
        auto prevReq = prevReqs.get(i);
        auto newReq = newReqs.get(i);

        if (prevReq == DimStride::Any &&
            newReq == DimStride::Any) {
            continue;
        }

        // In case if both requirements are defined, use `prevReq`.
        // We'll check that both requirements are satisfied at the end.
        if (prevReq != DimStride::Any) {
            mergedReqs.add(i, prevReq);
        } else {
            mergedReqs.add(i, newReq);
        }
    }

    _requiredStrides = mergedReqs;

    IE_ASSERT(checkStrides(prevReqs));
    IE_ASSERT(checkStrides(newReqs));
}

void DataNode::clearAllocation() {
    _location = DataLocation::None;
    _memoryOffset = 0;
    attrs().erase("ioBufferOffset");
}

void DataNode::setMemReqs(MemoryType mem) {
    if (mem != MemoryType::DDR) {
        IE_ASSERT(_usage == DataUsage::Intermediate);
    }

    _memReqs = mem;
}

void DataNode::setIOInfo(DataLocation location, int ioBufferOffset) {
    IE_ASSERT(_usage == DataUsage::Input || _usage == DataUsage::Output);

    if (_usage == DataUsage::Input) {
        IE_ASSERT(location == DataLocation::Input);
    } else if (_usage == DataUsage::Output) {
        IE_ASSERT(location == DataLocation::Output);
    }

    _location = location;
    _memoryOffset = 0;
    attrs().set<int>("ioBufferOffset", ioBufferOffset);
}

void DataNode::setAllocationInfo(DataLocation location, int memoryOffset) {
    IE_ASSERT(_usage == DataUsage::Const || _usage == DataUsage::Intermediate || _usage == DataUsage::Temp);

    if (_usage == DataUsage::Const) {
        IE_ASSERT(location == DataLocation::Blob);
    } else if (_usage == DataUsage::Temp) {
        IE_ASSERT(location == DataLocation::BSS);
    }

    _location = location;
    _memoryOffset = memoryOffset;
}

void DataNode::serializeNewBuffer(
        BlobSerializer& serializer,
        DimsOrder newOrder) {
    if (newOrder.numDims() == 0) {
        serializeBufferImpl(serializer, _desc, this->strides());
    } else {
        IE_ASSERT(newOrder.numDims() >= _desc.dimsOrder().numDims());

        auto newDims = _desc.dims();
        auto newStrides = this->strides();
        auto newPerm = newOrder.toPermutation();

        auto origOrder = _desc.dimsOrder();
        auto origPerm = origOrder.toPermutation();

        int origPermInd = 0;
        for (int i = 0; i < newPerm.size(); i++) {
            auto d = newPerm[i];

            if (origPermInd < origPerm.size() && origPerm[origPermInd] == d) {
                ++origPermInd;
                continue;
            }

            newDims.set(d, 1);
            if (i == 0) {
                newStrides.set(d, _desc.elemSize());
            } else {
                newStrides.set(d, newStrides[newPerm[i - 1]] * newDims[newPerm[i - 1]]);
            }
        }
        IE_ASSERT(origPermInd == origPerm.size());

        DataDesc newDesc(_desc.type(), newOrder, newDims);
        serializeBufferImpl(serializer, newDesc, newStrides);
    }
}

namespace {

// Decreases all order's valuable digits simultaneously so minimal digit is equal 1
void rebaseOrderToOne(DimsOrder& ord, DimValues& dims, DimValues& strides) {
    auto perm = ord.toPermutation();
    IE_ASSERT(!perm.empty());

    auto minDim = MAX_DIMS_64 + 1;
    for (auto d : perm) {
        minDim = std::min(minDim, static_cast<int>(d));
    }

    DimValues newDims;
    DimValues newStrides;

    for (int i = 0; i < perm.size(); ++i) {
        auto oldDim = perm[i];
        auto newDim = static_cast<Dim>(static_cast<int>(oldDim) - minDim);

        perm[i] = newDim;
        newDims.set(newDim, dims[oldDim]);
        newStrides.set(newDim, strides[oldDim]);
    }

    ord = DimsOrder::fromPermutation(perm);
    dims = newDims;
    strides = newStrides;
}

}  // namespace

void DataNode::serializeOldBuffer(
        const Stage& stage,
        BlobSerializer& serializer,
        DimsOrder newOrder,
        const EnumMap<Dim, std::vector<Dim>>& dimsReloc) {
    const int OLD_FORMAT_NUM_DIMS = 3;

    auto newDims = _desc.dims();
    auto newStrides = this->strides();

    //
    // Apply alternative DimsOrder if any.
    //

    if (newOrder.numDims() == 0) {
        newOrder = _desc.dimsOrder();
    } else {
        IE_ASSERT(newOrder.numDims() == OLD_FORMAT_NUM_DIMS);

        auto origPerm = _desc.dimsOrder().toPermutation();
        auto origIndeces = _desc.dimsOrder().toIndices();
        auto origDims = newDims;
        auto origStrides = newStrides;

        auto newPerm = newOrder.toPermutation();

        newDims.clear();
        newStrides.clear();

        //
        // Move real dims and strides according ro relocation map
        //

        EnumSet<Dim> usedOrigDims;
        int prevOrigDimInd = -1;

        for (int i = 0; i < newPerm.size(); ++i) {
            auto newDim = newPerm[i];

            int newDimVal = 1;
            int newStrideVal = 0;
            if (i == 0) {
                newStrideVal = _desc.elemSize();
            } else {
                newStrideVal = newStrides[newPerm[i - 1]] * newDims[newPerm[i - 1]];
            }

            auto it = dimsReloc.find(newDim);
            if (it != dimsReloc.end()) {
                auto origDimsToReloc = it->second;
                IE_ASSERT(!origDimsToReloc.empty());

                for (int j = 0; j < origDimsToReloc.size(); ++j) {
                    auto origDim = origDimsToReloc[j];
                    auto origDimInd = origIndeces[origDim];

                    IE_ASSERT(usedOrigDims.count(origDim) == 0);
                    IE_ASSERT(_desc.dimsOrder().hasDim(origDim));
                    IE_ASSERT(origDimInd == prevOrigDimInd + 1);

                    usedOrigDims.insert(origDim);

                    if (j > 0 && origDims[origDim] > 1) {
                        IE_ASSERT(checkStride(origStrides, _desc, origDimInd, DimStride::Compact));
                    }

                    newDimVal *= origDims[origDim];
                    if (j == 0) {
                        newStrideVal = origStrides[origDim];
                    }

                    prevOrigDimInd = origDimInd;
                }
            }

            newDims.set(newDim, newDimVal);
            newStrides.set(newDim, newStrideVal);
        }

        IE_ASSERT(usedOrigDims.size() == origDims.size());
        for (auto usedDim : usedOrigDims) {
            IE_ASSERT(_desc.dimsOrder().hasDim(usedDim));
        }
    }

    //
    // Adjust num dims and dims order to FixedNumDims
    //

    auto newPerm = newOrder.toPermutation();
    IE_ASSERT(!newPerm.empty());

    int maxDimDigit = -1;
    for (auto d : newPerm) {
        maxDimDigit = std::max(maxDimDigit, static_cast<int>(d));
    }
    IE_ASSERT(maxDimDigit >= 0);

    if (newPerm.size() < OLD_FORMAT_NUM_DIMS) {
        for (int i = newPerm.size(); i < OLD_FORMAT_NUM_DIMS; i++) {
            auto lastDim = newPerm.back();
            auto newLastDim = static_cast<Dim>(++maxDimDigit);

            newDims.set(newLastDim, 1);
            newStrides.set(newLastDim, newStrides[lastDim] * newDims[lastDim]);

            newPerm.emplace_back(newLastDim);
        }

        newOrder = DimsOrder::fromPermutation(newPerm);
    }

    if (newPerm.size() > OLD_FORMAT_NUM_DIMS) {
        for (int i = OLD_FORMAT_NUM_DIMS; i < newPerm.size(); i++) {
            IE_ASSERT(newDims[newPerm[i]] == 1);
            newDims.erase(newPerm[i]);
            newStrides.erase(newPerm[i]);
        }

        newPerm.resize(OLD_FORMAT_NUM_DIMS);

        newOrder = DimsOrder::fromPermutation(newPerm);
    }

    rebaseOrderToOne(newOrder, newDims, newStrides);

    IE_ASSERT(newOrder.numDims() == OLD_FORMAT_NUM_DIMS);
    IE_ASSERT(newOrder == DimsOrder::HWC || newOrder == DimsOrder::CHW || newOrder == DimsOrder::HCW);

    //
    // Create new DataDesc
    //

    DataDesc newDesc(_desc.type(), newOrder, newDims);

    if (stage != nullptr) {
        for (const auto& inEdge : stage->inputEdges()) {
            if (inEdge->input() == handle_from_this()) {
                inEdge->attrs().set<DataDesc>("newDesc", newDesc);
                inEdge->attrs().set<DimValues>("newStrides", newStrides);
            }
        }
        for (const auto& outEdge : stage->outputEdges()) {
            if (outEdge->output() == handle_from_this()) {
                outEdge->attrs().set<DataDesc>("newDesc", newDesc);
                outEdge->attrs().set<DimValues>("newStrides", newStrides);
            }
        }
    }

    //
    // Serialize update data
    //

    serializeBufferImpl(serializer, newDesc, newStrides);
}

void DataNode::serializeIOInfo(BlobSerializer& serializer) const {
    auto ioIdx = attrs().get<int>("ioIdx");
    serializer.append(checked_cast<uint32_t>(ioIdx));

    auto ioBufferOffset = attrs().get<int>("ioBufferOffset");
    serializer.append(checked_cast<uint32_t>(ioBufferOffset));

    auto nameLength = checked_cast<uint32_t>(_name.length());
    auto nameLengthAligned = alignVal(nameLength, 16u);

    serializer.append(nameLengthAligned);
    for (auto c : _name) {
        serializer.append(c);
    }
    for (uint32_t i = 0; i < nameLengthAligned - nameLength; ++i) {
        serializer.append(uint8_t(0));
    }

    serializeDescImpl(serializer, _desc, strides());
}

void DataNode::serializeDescImpl(
        BlobSerializer& serializer,
        const DataDesc& storedDesc,
        const DimValues& storedStrides) const {
    IE_ASSERT(storedDesc.numDims() <= MAX_DIMS_32);

    const auto& storedDims = storedDesc.dims();

    auto storedDimsOrder = storedDesc.dimsOrder();

    auto storedPerm = storedDimsOrder.toPermutation();
    IE_ASSERT(!storedPerm.empty());

    serializer.append(checked_cast<uint32_t>(storedDesc.type()));
    serializer.append(checked_cast<uint32_t>(storedDimsOrder.code()));

    serializer.append(checked_cast<uint32_t>(storedPerm.size()));
    for (auto d : storedPerm) {
        serializer.append(checked_cast<uint32_t>(storedDims[d]));
    }
    for (auto d : storedPerm) {
        serializer.append(checked_cast<uint32_t>(storedStrides[d]));
    }
}

void DataNode::serializeBufferImpl(
        BlobSerializer& serializer,
        const DataDesc& storedDesc,
        const DimValues& storedStrides) const {
    serializeDescImpl(serializer, storedDesc, storedStrides);

    serializer.append(checked_cast<uint32_t>(_location));

    if (_location == DataLocation::Input || _location == DataLocation::Output) {
        auto topParent = getTopParentData();

        auto ioIdx = topParent->attrs().get<int>("ioIdx");
        serializer.append(checked_cast<uint32_t>(ioIdx));

        auto parentByteSize = topParent->totalByteSize();
        serializer.append(checked_cast<uint32_t>(parentByteSize));
    }

    serializer.append(checked_cast<uint32_t>(_memoryOffset));
}

void printTo(std::ostream& os, const Data& data) {
    os << (data == nullptr ? "<null>" : data->name());
}

//
// loopOverData
//

namespace {

struct StopSignal final {};

void loopOverDataImpl(
        const Data& data,
        const FuncRef<DataLoopStatus(const Data&)>& op) {
    for (const auto& childData : data->childDatas()) {
        auto status = op(childData);

        if (status == DataLoopStatus::NextChild) {
            loopOverDataImpl(childData, op);
        } else if (status == DataLoopStatus::Stop) {
            throw StopSignal();
        }
    }
}

}  // namespace

void loopOverData(
        const Data& data,
        const FuncRef<DataLoopStatus(const Data&)>& op) {
    auto status = op(data);
    if (status != DataLoopStatus::NextChild)
        return;

    try {
        loopOverDataImpl(data, op);
    } catch (const StopSignal&) {
        return;
    }
}

}  // namespace vpu
