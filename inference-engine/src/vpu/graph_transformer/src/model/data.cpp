// Copyright (C) 2018-2020 Intel Corporation
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
#include <utility>

#include <precision_utils.h>
#include <ie_parallel.hpp>

#include <vpu/model/edges.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/backend/backend.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

//
// DataContent
//

DataContent::~DataContent() = default;

const void* CalculatedDataContent::getRaw() const {
    if (_temp.empty()) {
        _temp.resize(getTempBufSize(_baseContents));
        fillTempBuf(_baseContents, _temp.data());
        _baseContents.clear();
    }
    return _temp.data();
}

size_t CalculatedDataContent::getTempBufSize(const SmallVector<DataContent::Ptr, 2>&) const {
    return checked_cast<size_t>(desc().totalDimSize()) *
           checked_cast<size_t>(desc().elemSize());
}

namespace {

class IeBlobContent final : public DataContent {
public:
    IeBlobContent(const ie::Blob::Ptr& blob, int repeat) : _blob(blob), _repeat(repeat) {}

protected:
    const void* getRaw() const override {
        if (desc().type() == DataType::FP16) {
            if (_blobFp16 == nullptr) {
                _blobFp16 = getBlobFP16(_blob);
                _blob.reset();
            }

            if (_repeat == 1) {
                return _blobFp16->cbuffer();
            } else {
                if (_tempFp16.empty()) {
                    VPU_PROFILE(IeBlobContent);

                    IE_ASSERT(desc().totalDimSize() % _repeat == 0);

                    auto origNumElems = desc().totalDimSize() / _repeat;
                    IE_ASSERT(checked_cast<size_t>(origNumElems) <= _blobFp16->size());

                    auto origPtr = _blobFp16->cbuffer().as<const fp16_t*>();
                    IE_ASSERT(origPtr != nullptr);

                    _tempFp16.resize(checked_cast<size_t>(desc().totalDimSize()));

                    ie::parallel_for(_repeat, [this, origPtr, origNumElems](int i) {
                        std::copy_n(origPtr, origNumElems, _tempFp16.data() + i * origNumElems);
                    });
                }

                return _tempFp16.data();
            }
        } else if (desc().type() == DataType::S32) {
            if (_repeat == 1) {
                return _blob->cbuffer();
            } else {
                if (_tempS32.empty()) {
                    VPU_PROFILE(IeBlobContent);

                    IE_ASSERT(desc().totalDimSize() % _repeat == 0);

                    auto origNumElems = desc().totalDimSize() / _repeat;
                    IE_ASSERT(checked_cast<size_t>(origNumElems) <= _blob->size());

                    auto origPtr = _blob->cbuffer().as<const int32_t*>();
                    IE_ASSERT(origPtr != nullptr);

                    _tempS32.resize(checked_cast<size_t>(desc().totalDimSize()));

                    ie::parallel_for(_repeat, [this, origPtr, origNumElems](int i) {
                        std::copy_n(origPtr, origNumElems, _tempS32.data() + i * origNumElems);
                    });
                }

                return _tempS32.data();
            }
        } else {
            VPU_THROW_EXCEPTION << "Unsupported data type " << desc().type();
        }
    }

private:
    mutable ie::Blob::Ptr _blob;
    int _repeat = 0;

    mutable ie::Blob::Ptr _blobFp16;
    mutable std::vector<fp16_t> _tempFp16;
    mutable std::vector<int32_t> _tempS32;
};

}  // namespace

DataContent::Ptr ieBlobContent(const ie::Blob::Ptr& blob, int repeat) {
    return std::make_shared<IeBlobContent>(blob, repeat);
}

namespace {

class ReplicatedContent final : public CalculatedDataContent {
public:
    ReplicatedContent(float val, int count) : _factor{val}, _count(count) {}

    ReplicatedContent(DataContent::Ptr origContent, int count) :
        CalculatedDataContent({std::move(origContent)}), _count(count) {
    }

protected:
    size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>& baseContents) const override {
        if (baseContents.empty()) {
            return checked_cast<size_t>(_count) * sizeof(fp16_t);
        } else {
            IE_ASSERT(baseContents.size() == 1);
            IE_ASSERT(desc().totalDimSize() % _count == 0);

            return checked_cast<size_t>(desc().totalDimSize()) * sizeof(fp16_t);
        }
    }

    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(ReplicatedContent);

        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        if (baseContents.empty()) {
            std::fill_n(dstPtr, _count, ie::PrecisionUtils::f32tof16(_factor));
        } else {
            IE_ASSERT(baseContents.size() == 1);
            IE_ASSERT(desc().totalDimSize() % _count == 0);

            auto origCount = desc().totalDimSize() / _count;
            auto origPtr = baseContents[0]->get<fp16_t>();
            IE_ASSERT(origPtr != nullptr);

            ie::parallel_for(_count, [origPtr, origCount, dstPtr](int i) {
                std::copy_n(origPtr, origCount, dstPtr + i * origCount);
            });
        }
    }

private:
    float _factor = 1.0f;
    int _count = 0;
};

}  // namespace

DataContent::Ptr replicateContent(float val, int count) {
    return std::make_shared<ReplicatedContent>(val, count);
}

DataContent::Ptr replicateContent(const DataContent::Ptr& origContent, int count) {
    return std::make_shared<ReplicatedContent>(origContent, count);
}

namespace {

class ScaledContent final : public CalculatedDataContent {
public:
    ScaledContent(const DataContent::Ptr& origContent, float scale) :
        CalculatedDataContent({origContent}), _factor(scale) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(ScaledContent);

        IE_ASSERT(baseContents.size() == 1);

        auto totalSize = desc().totalDimSize();

        auto origDesc = baseContents[0]->desc();
        IE_ASSERT(origDesc.type() == DataType::FP16);
        IE_ASSERT(origDesc.totalDimSize() == totalSize);

        auto srcPtr = baseContents[0]->get<fp16_t>();
        IE_ASSERT(srcPtr != nullptr);

        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        ie::parallel_for(totalSize, [this, srcPtr, dstPtr](int i) {
            dstPtr[i] = ie::PrecisionUtils::f32tof16(ie::PrecisionUtils::f16tof32(srcPtr[i]) * _factor);
        });
    }

private:
    float _factor = 1.0f;
};

}  // namespace

DataContent::Ptr scaleContent(const DataContent::Ptr& origContent, float scale) {
    return std::make_shared<ScaledContent>(origContent, scale);
}

namespace {

class ScaledChannelContent final : public CalculatedDataContent {
public:
    ScaledChannelContent(
            const DataContent::Ptr& origContent,
            const DataContent::Ptr& scaleContent) :
            CalculatedDataContent({origContent, scaleContent}) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(ScaledChannelContent);

        IE_ASSERT(baseContents.size() == 2);

        auto totalSize = desc().totalDimSize();

        IE_ASSERT(desc().numDims() == 4 && desc().dimsOrder() == DimsOrder::NCHW);
        auto numN = desc().dim(Dim::N);
        auto numC = desc().dim(Dim::C);
        auto numH = desc().dim(Dim::H);
        auto numW = desc().dim(Dim::W);

        auto origDesc = baseContents[0]->desc();
        IE_ASSERT(origDesc.type() == DataType::FP16);
        IE_ASSERT(origDesc.totalDimSize() == totalSize);
        IE_ASSERT(baseContents[1]->desc().totalDimSize() == numN);

        auto srcPtr = baseContents[0]->get<fp16_t>();
        IE_ASSERT(srcPtr != nullptr);

        auto scale = baseContents[1]->get<fp16_t>();
        IE_ASSERT(scale != nullptr);

        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        for (int n = 0; n < numN; n++) {
            for (int c = 0; c < numC; c++) {
               for (int h = 0; h < numH; h++) {
                   for (int w = 0; w < numW; w++) {
                       dstPtr[n * numC * numH * numW + c * numH * numW + h * numW + w] =
                               srcPtr[n * numC * numH * numW + c * numH * numW + h * numW + w] * scale[n];
                   }
               }
            }
        }
    }
};

}  // namespace

DataContent::Ptr scaledChannelContent(
        const DataContent::Ptr& origContent,
        const DataContent::Ptr& scaleContent) {
    return std::make_shared<ScaledChannelContent>(origContent, scaleContent);
}

//
// DataNode
//

Data DataNode::getTopParentData() const {
    Data topParent = this;
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

bool DataNode::canHaveAParent() const {
    return parentData() == nullptr && usage() == DataUsage::Intermediate;
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
    const auto& fixedRequirements = prevReqs.fixedStrides().empty() ? newReqs : prevReqs;
    if (!fixedRequirements.fixedStrides().empty()) {
        mergedReqs = fixedRequirements;
    } else {
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

void DataNode::serializeBuffer(
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

        size_t origPermInd = 0;
        for (size_t i = 0; i < newPerm.size(); i++) {
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
