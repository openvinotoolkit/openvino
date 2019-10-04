// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <functional>
#include <vector>

#include <ie_data.h>
#include <ie_blob.h>

#include <vpu/model/base.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/model/data_desc.hpp>
#include <vpu/backend/blob_serializer.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/func_ref.hpp>

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
// DataLocation
//

//
// Describes where Data object is located.
//

// Must be synchronized with MvTensor
VPU_DECLARE_ENUM(DataLocation,
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

//
// DataContent
//

//
// Content of the Const Data object.
//

class DataContent {
public:
    using Ptr = std::shared_ptr<DataContent>;

    virtual ~DataContent() = default;

    // TYPED pointer
    template <typename T>
    const T* get() const {
        return static_cast<const T*>(getRaw());
    }

    const DataDesc& desc() const { return _desc; }

protected:
    // RAW pointer
    virtual const void* getRaw() const = 0;

protected:
    DataDesc _desc;
    friend class Model;
};

//
// Data content that is calculated on the fly, using lazy calculation:
//
//   * It performs calculation on the first call and stores it in internal buffer.
//   * Next access will return the pointer to calculated buffer.
//
class CalculatedDataContent : public DataContent {
public:
    CalculatedDataContent() = default;
    explicit CalculatedDataContent(std::initializer_list<DataContent::Ptr> baseContents) : _baseContents(baseContents) {}

protected:
    const void* getRaw() const override;

    virtual size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>& baseContents) const;
    virtual void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const = 0;

private:
    mutable SmallVector<DataContent::Ptr, 2> _baseContents;
    mutable std::vector<uint8_t> _temp;
};

DataContent::Ptr ieBlobContent(
        const ie::Blob::Ptr& blob,
        int repeat = 1);

DataContent::Ptr replicateContent(
        float val,
        int count);

DataContent::Ptr replicateContent(
        const DataContent::Ptr& origContent,
        int count);

DataContent::Ptr scaleContent(
        const DataContent::Ptr& origContent,
        float scale);

//
// DataNode
//

class DataNode final :
        public EnableHandleFromThis<DataNode>,
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
    VPU_MODEL_ATTRIBUTE(SharedAllocation, parentDataEdge, nullptr)

    /**
     * Children data edges uses parent's memory
     */
    VPU_MODEL_ATTRIBUTE_PTR_RANGE(SharedAllocationList, childDataEdges)

    //
    // Const data content
    //

    VPU_MODEL_ATTRIBUTE(DataContent::Ptr, content, nullptr)

    //
    // Allocation info
    //

    VPU_MODEL_ATTRIBUTE(MemoryType, memReqs, MemoryType::DDR)
    VPU_MODEL_ATTRIBUTE(DataLocation, location, DataLocation::None)
    VPU_MODEL_ATTRIBUTE(int, memoryOffset, 0)

    //
    // Edges wrappers
    //

private:
    struct ConsumerAccess final {
        inline auto operator()(const StageInput& edge) const -> decltype(edge->consumer()) {
            return edge->consumer();
        }
    };

    struct ChildDataAccess final {
        inline auto operator()(const SharedAllocation& edge) const -> decltype(edge->child()) {
            return edge->child();
        }
    };

public:
    inline Stage producer() const {
        return _producerEdge == nullptr ? nullptr : _producerEdge->producer();
    }

    inline int numConsumers() const {
        return _consumerEdges.size();
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
        return _parentDataEdge == nullptr ? nullptr : _parentDataEdge->parent();
    }

    inline int numChildDatas() const {
        return _childDataEdges.size();
    }
    inline auto childDatas() const -> decltype(mapRange<ChildDataAccess>(childDataEdges())) {
        return mapRange<ChildDataAccess>(childDataEdges());
    }

    Data getTopParentData() const;

    //
    // DataDesc
    //

    DimValues strides() const;

    int totalByteSize() const;

    int elemOffset(const DimValues& coord) const;
    int lastElemOffset() const;

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

    void setIOInfo(DataLocation location, int ioBufferOffset);

    void setAllocationInfo(DataLocation location, int memoryOffset);

    //
    // Backend utilities
    //

    // Serialize as-is for new MvTensor kernels that can work with ND data.
    // If `newOrder` is not empty, it will be used instead of original and missing dimensions will be set to 1.
    void serializeNewBuffer(
            BlobSerializer& serializer,
            DimsOrder newOrder = DimsOrder());

    // Serialize for deprecated MvTensor kernels that can work only with 3D data.
    //
    // `dimsReloc` is a map from new dims to original dims.
    // Empty record means use 1 for the new dim and reuse previous stride.
    // For example :
    //   * Original order : NC
    //   * `newOrder` : HWC
    //   * `dimsReloc` : {(C -> C), {H -> N}}
    // The Data will be serialized as HWC with
    //   * newDims[H] == origDims[N]
    //   * newDims[W] == 1
    //   * newDims[C] == origDims[C]
    // If there is several original dims per new dim, they will be multiplied
    // (assuming that original dims are near and have no strides between).
    void serializeOldBuffer(
            const Stage& stage,
            BlobSerializer& serializer,
            DimsOrder newOrder = DimsOrder(),
            const EnumMap<Dim, DimVector>& dimsReloc = EnumMap<Dim, DimVector>());

    void serializeOldBufferNC(
            const Stage& stage,
            BlobSerializer& serializer);


    void serializeIOInfo(BlobSerializer& serializer) const;

private:
    void serializeDescImpl(
            BlobSerializer& serializer,
            const DataDesc& storedDesc,
            const DimValues& storedStrides) const;

    void serializeBufferImpl(
            BlobSerializer& serializer,
            const DataDesc& storedDesc,
            const DimValues& storedStrides) const;

private:
    inline DataNode() :
        _consumerEdges(&StageInputEdge::_posInData),
        _childDataEdges(&SharedAllocationEdge::_posInData),
        _posInModel(this) {
    }

private:
    Handle<Model> _model;
    DataPtrList::iterator _ptrPosInModel;
    IntrusivePtrListNode<DataNode> _posInModel;

    friend class Model;
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
