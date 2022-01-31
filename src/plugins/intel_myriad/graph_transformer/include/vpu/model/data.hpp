// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/base.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/model/data_desc.hpp>
#include <vpu/model/data_contents/data_content.hpp>
#include <vpu/backend/blob_serializer.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/func_ref.hpp>

#include <ie_data.h>
#include <ie_blob.h>

#include <memory>
#include <string>
#include <functional>
#include <vector>

namespace vpu {

namespace ie = InferenceEngine;

//
// DataUsage
//

//
// Describes how Data object is used in the Model:
//   * Input / Output : network input or output.
//   * Const : constant values (weights, biases, etc.).
//   * Intermediate : Data that are used for intermediate results.
//   * Temp : temporary buffer.
//   * Fake : fake Data object to fill stage input/output port.
//

VPU_DECLARE_ENUM(DataUsage,
    Input,
    Output,
    Const,
    Intermediate,
    Temp,
    Fake
)

//
// Location
//

//
// Describes where particular data or shape is located.
//

// Must be synchronized with MvTensor
VPU_DECLARE_ENUM(Location,
    None = 0,
    Input = 1,
    Output = 2,
    Blob = 3,
    BSS = 4,
    CMX = 5
)

VPU_DECLARE_ENUM(MemoryType,
    DDR,
    CMX)

struct DataLocation final {
    Location location;
    int offset;
};

static constexpr DataLocation defaultDataLocation = {
    Location::None, 0
};

struct ShapeLocation final {
    Location dimsLocation;
    int dimsOffset;
    Location stridesLocation;
    int stridesOffset;

    bool operator==(const ShapeLocation& shapeLocation) const {
        return std::tie(dimsLocation, dimsOffset, stridesLocation, stridesOffset) ==
               std::tie(shapeLocation.dimsLocation, shapeLocation.dimsOffset, shapeLocation.stridesLocation, shapeLocation.stridesOffset);
    }

    bool operator!=(const ShapeLocation& shapeLocation) const {
        return !(*this == shapeLocation);
    }
};

static constexpr ShapeLocation defaultShapeLocation = {
    Location::None, 0, Location::None, 0
};

//
// DataNode
//

class DataNode final :
        public EnableHandle,
        public EnableCustomAttributes {
    //
    // Main attributes
    //

    VPU_MODEL_ATTRIBUTE(std::string, name, std::string())
    VPU_MODEL_ATTRIBUTE(DataUsage, usage, DataUsage::Fake)
    VPU_MODEL_ATTRIBUTE(DataDesc, desc, DataDesc())
    VPU_MODEL_ATTRIBUTE(StridesRequirement, requiredStrides, StridesRequirement::empty())

    //
    // Bindings with IE
    //

    VPU_MODEL_ATTRIBUTE(ie::DataPtr, origData, nullptr)

    //
    // Edges
    //

    VPU_MODEL_ATTRIBUTE(StageOutput, producerEdge, nullptr)
    VPU_MODEL_ATTRIBUTE_PTR_RANGE(StageInputList, consumerEdges)

    VPU_MODEL_ATTRIBUTE(StageTempBuffer, tempBufferEdge, nullptr)

    /**
     * Parent data edge actually allocates memory
     */
    VPU_MODEL_ATTRIBUTE(DataToDataAllocation, parentDataToDataEdge, nullptr)

    /**
     * Children data edges uses parent's memory
     */
    VPU_MODEL_ATTRIBUTE_PTR_RANGE(DataToDataAllocationList, childDataToDataEdges)

    /**
     * Parent data edge actually allocates memory as a shape for current data
     */
    VPU_MODEL_ATTRIBUTE(DataToShapeAllocation, parentDataToShapeEdge, nullptr)

    /**
     * Children data edges uses parent's memory as a shape
     */
    VPU_MODEL_ATTRIBUTE_PTR_RANGE(DataToShapeAllocationList, childDataToShapeEdges)

    //
    // Const data content
    //

    VPU_MODEL_ATTRIBUTE(DataContent::Ptr, content, nullptr)

    //
    // Allocation info
    //

    VPU_MODEL_ATTRIBUTE(MemoryType, memReqs, MemoryType::DDR)
    VPU_MODEL_ATTRIBUTE(DataLocation, dataLocation, defaultDataLocation)
    VPU_MODEL_ATTRIBUTE(ShapeLocation, shapeLocation, defaultShapeLocation)

    //
    // Edges wrappers
    //

    VPU_MODEL_ATTRIBUTE(Model, model, nullptr)

private:
    struct ConsumerAccess final {
        inline auto operator()(const StageInput& edge) const -> decltype(edge->consumer()) {
            return edge->consumer();
        }
    };

    struct ChildDataAccess final {
        inline auto operator()(const DataToDataAllocation& edge) const -> decltype(edge->child()) {
            return edge->child();
        }
    };

public:
    inline Stage producer() const {
        return _producerEdge == nullptr ? nullptr : _producerEdge->producer();
    }

    inline int numConsumers() const {
        return static_cast<int>(_consumerEdges.size());
    }
    inline auto consumers() const -> decltype(mapRange<ConsumerAccess>(consumerEdges())) {
        return mapRange<ConsumerAccess>(consumerEdges());
    }
    inline StageInput singleConsumerEdge() const {
        IE_ASSERT(_consumerEdges.size() == 1);
        return *_consumerEdges.begin();
    }
    inline Stage singleConsumer() const {
        return singleConsumerEdge()->consumer();
    }

    inline Data parentData() const {
        return _parentDataToDataEdge == nullptr ? nullptr : _parentDataToDataEdge->parent();
    }

    inline int numChildDatas() const {
        return static_cast<int>(_childDataToDataEdges.size());
    }
    inline auto childDatas() const -> decltype(mapRange<ChildDataAccess>(childDataToDataEdges())) {
        return mapRange<ChildDataAccess>(childDataToDataEdges());
    }

    Data getTopParentData() const;

    bool isConsumed() const;

    //
    // DataDesc
    //

    DimValues strides() const;

    int totalByteSize() const;

    int elemOffset(const DimValues& coord) const;
    int lastElemOffset() const;

    bool canHaveAParent() const;

    //
    // Bindings with IE
    //

    inline void setOrigData(const ie::DataPtr& origData) { _origData = origData; }

    //
    // StridesRequirement
    //

    bool checkStrides(const StridesRequirement& reqs) const;

    inline void resetRequiredStrides() {
        _requiredStrides = StridesRequirement::empty();
    }

    void updateRequiredStrides(const StridesRequirement& newReqs);

    //
    // Allocation info
    //

    void clearAllocation();

    void setMemReqs(MemoryType mem);

    void setIOInfo(Location location, int ioBufferOffset);

    void setDataAllocationInfo(const DataLocation& dataLocation);

    void setShapeAllocationInfo(const ShapeLocation& shapeLocation);

    bool isShapeAllocated() const;

    //
    // Backend utilities
    //

    // Serialize as-is for new MvTensor kernels that can work with ND data.
    void serializeBuffer(BlobSerializer& serializer);

    void serializeIOInfo(BlobSerializer& serializer) const;

private:
    void serializeDescImpl(
            BlobSerializer& serializer,
            const DataDesc& storedDesc,
            const ShapeLocation& shapeLocation) const;

private:
    inline DataNode() :
        _consumerEdges(&StageInputEdge::_posInData),
        _childDataToDataEdges(&DataToDataAllocationEdge::_posInData),
        _childDataToShapeEdges(&DataToShapeAllocationEdge::_posInData),
        _posInModel(this) {
    }

private:
    DataPtrList::iterator _ptrPosInModel;
    DataListNode _posInModel;

    friend ModelObj;
};

void printTo(std::ostream& os, const Data& data);

//
// loopOverData
//

VPU_DECLARE_ENUM(DataLoopStatus,
    NextChild,
    NextSibling,
    Stop)

void loopOverData(
        const Data& data,
        const FuncRef<DataLoopStatus(const Data&)>& op);

}  // namespace vpu
