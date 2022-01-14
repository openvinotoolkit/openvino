// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data.hpp>

#include <vpu/model/edges.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/backend/backend.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/compile_env.hpp>

#include <precision_utils.h>

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

namespace vpu {

//
// DataNode
//

bool DataNode::isConsumed() const {
    return numConsumers() > 0 || !childDataToShapeEdges().empty();
}

Data DataNode::getTopParentData() const {
    Data topParent = this;
    while (auto nextParent = topParent->parentData()) {
        topParent = nextParent;
    }
    return topParent;
}

DimValues DataNode::strides() const {
    if (_parentDataToDataEdge != nullptr) {
        if (_parentDataToDataEdge->mode() == SharedDataMode::ROI) {
            return _parentDataToDataEdge->parent()->strides();
        }
    }

    return calcStrides(_desc, _requiredStrides);
}

int DataNode::totalByteSize() const {
    // IT doesn't have sence for child Data.
    IE_ASSERT(_parentDataToDataEdge == nullptr);

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
    IE_ASSERT(_parentDataToDataEdge == nullptr);
    IE_ASSERT(_childDataToDataEdges.empty());

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
    _dataLocation = defaultDataLocation;
    attrs().erase("ioBufferOffset");
}

void DataNode::setMemReqs(MemoryType mem) {
    if (mem != MemoryType::DDR) {
        IE_ASSERT(_usage == DataUsage::Intermediate);
    }

    _memReqs = mem;
}

void DataNode::setIOInfo(Location location, int ioBufferOffset) {
    VPU_INTERNAL_CHECK(_usage == DataUsage::Input || _usage == DataUsage::Output,
        "Data {} failed: setIOInfo called for non IO data, actual usage is {}",
        name(), usage());

    if (_usage == DataUsage::Input) {
        VPU_INTERNAL_CHECK(location == Location::Input,
            "Input data {} failed: setIOInfo called with non input location, actual location is {}",
            name(), location);
    } else if (_usage == DataUsage::Output) {
        VPU_INTERNAL_CHECK(location == Location::Output,
            "Output data {} failed: setIOInfo called with non output location, actual location is {}",
            name(), location);
    }

    _dataLocation = {location, 0};
    attrs().set<int>("ioBufferOffset", ioBufferOffset);
}

void DataNode::setDataAllocationInfo(const DataLocation& dataLocation) {
    VPU_INTERNAL_CHECK(_usage == DataUsage::Const || _usage == DataUsage::Intermediate || _usage == DataUsage::Temp,
        "Data {} failed: setDataAllocationInfo called for data with incorrect usage, actual usage: {} "
        "valid usages: {}, {}, {}", name(), usage(), DataUsage::Const, DataUsage::Intermediate, DataUsage::Temp);

    if (_usage == DataUsage::Const) {
        VPU_INTERNAL_CHECK(dataLocation.location == Location::Blob,
            "Const data {} failed: setDataAllocationInfo called with non blob location, actual location is {}",
            name(), dataLocation.location);
    } else if (_usage == DataUsage::Temp) {
        VPU_INTERNAL_CHECK(dataLocation.location == Location::BSS,
            "Temp data {} failed: setDataAllocationInfo called with non bss location, actual location is {}",
            name(), dataLocation.location);
    }

    _dataLocation = dataLocation;
}

void DataNode::setShapeAllocationInfo(const ShapeLocation& shapeLocation) {
    _shapeLocation = shapeLocation;
}

bool DataNode::isShapeAllocated() const {
    return _shapeLocation != defaultShapeLocation;
}

void DataNode::serializeBuffer(
        BlobSerializer& serializer) {
    serializeDescImpl(serializer, _desc, this->shapeLocation());

    serializer.append(checked_cast<uint32_t>(_dataLocation.location));

    const auto serializeIOParams = [&serializer](const Data& parent) {
        auto IOIdx = parent->attrs().get<int>("ioIdx");
        serializer.append(checked_cast<uint32_t>(IOIdx));

        auto parentByteSize = parent->totalByteSize();
        serializer.append(checked_cast<uint32_t>(parentByteSize));
    };

    if (_dataLocation.location == Location::Input || _dataLocation.location == Location::Output) {
        serializeIOParams(getTopParentData());
    }

    if (_shapeLocation.dimsLocation == Location::Output) {
        serializeIOParams(parentDataToShapeEdge()->parent());
    }

    if (_shapeLocation.stridesLocation == Location::Output) {
        serializeIOParams(parentDataToShapeEdge()->parent());
    }

    serializer.append(checked_cast<uint32_t>(_dataLocation.offset));
}

void DataNode::serializeIOInfo(BlobSerializer& serializer) const {
    auto dataIOIdx = attrs().get<int>("ioIdx");
    serializer.append(checked_cast<uint32_t>(dataIOIdx));

    auto ioBufferOffset = attrs().get<int>("ioBufferOffset");
    serializer.append(checked_cast<uint32_t>(ioBufferOffset));

    auto nameLength = checked_cast<uint32_t>(_name.length());
    auto nameSize = nameLength + 1; // required to support c-string when the name length is multiple of 16
    auto nameSizeAligned = alignVal(nameSize, 16u);

    serializer.append(nameSizeAligned);
    for (auto c : _name) {
        serializer.append(c);
    }
    for (uint32_t i = 0; i < nameSizeAligned - nameLength; ++i) {
        serializer.append(uint8_t(0));
    }

    auto resShapeLocation = shapeLocation();
    if (resShapeLocation.dimsLocation != Location::Blob) {
        auto ioDimsUpperBoundOffset = attrs().get<int>("ioDimsUpperBoundOffset");
        resShapeLocation.dimsLocation = Location::Blob;
        resShapeLocation.dimsOffset = ioDimsUpperBoundOffset;
    }
    if (resShapeLocation.stridesLocation != Location::Blob) {
        auto ioStridesUpperBoundOffset = attrs().get<int>("ioStridesUpperBoundOffset");
        resShapeLocation.stridesLocation = Location::Blob;
        resShapeLocation.stridesOffset = ioStridesUpperBoundOffset;
    }

    serializeDescImpl(serializer, _desc, resShapeLocation);
}

void DataNode::serializeDescImpl(
        BlobSerializer& serializer,
        const DataDesc& storedDesc,
        const ShapeLocation& shapeLocation) const {
    IE_ASSERT(storedDesc.numDims() <= MAX_DIMS_32);

    auto storedDimsOrder = storedDesc.dimsOrder();

    auto storedPerm = storedDimsOrder.toPermutation();
    IE_ASSERT(!storedPerm.empty());

    serializer.append(checked_cast<uint32_t>(storedDesc.type()));
    serializer.append(checked_cast<uint32_t>(storedDimsOrder.code()));

    serializer.append(checked_cast<uint32_t>(storedPerm.size()));

    serializer.append(checked_cast<uint32_t>(shapeLocation.dimsLocation));
    serializer.append(checked_cast<uint32_t>(shapeLocation.dimsOffset));
    serializer.append(checked_cast<uint32_t>(shapeLocation.stridesLocation));
    serializer.append(checked_cast<uint32_t>(shapeLocation.stridesOffset));
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
