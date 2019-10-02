// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <set>

#include <ie_layers.h>

#include <vpu/model/base.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/data_desc.hpp>
#include <vpu/backend/blob_serializer.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/optional.hpp>

namespace vpu {

//
// StageType
//

// Must be synchronized with MvTensor
VPU_DECLARE_ENUM(StageType,
    //
    // This is special operations, that are not present in MvTensor,
    // but are used for internal Model processing and optimization.
    //

    __SPECIAL_START__ = -100000,

    //
    // Stages that have both HW and SW variants.
    // This is stubs that will be replaced with concrete implementation during the model optimization.
    //

    StubConv,
    StubMaxPool,
    StubAvgPool,
    StubFullyConnected,
    StubDeconv,

    Concat,
    Split,
    Reshape,
    Expand,
    Shrink,
    StridedSlice,

    Empty = -1,

    //
    // Normal operations
    //

    Conv = 0,
    MaxPool = 1,
    AvgPool = 2,
    SoftMax = 3,
    FC = 4,
    None = 5,
    Relu = 6,
    DepthConv = 8,
    Bias = 9,
    PRelu = 10,
    LRN = 11,
    Sum = 12,
    Prod = 13,
    Max = 14,
    Scale = 15,
    InnerLRN = 18,
    Copy = 19,
    Sigmoid = 20,
    Tanh = 21,
    Deconvolution = 22,
    Elu = 23,
    Power = 26,
    Crop = 27,
    Tile = 28,
    RegionYolo = 29,
    ReorgYolo = 30,
    Convert_u8f16 = 31,
    Convert_f32f16 = 32,
    Convert_f16f32 = 33,
    Permute = 34,
    Normalize = 35,
    DetectionOutput = 37,
    MyriadXHwOp = 38,
    CTCDecoder = 43,
    LeakyRelu = 44,
    BiasRelu = 45,
    BiasLeakyRelu = 46,
    ScaleShift = 47,
    Im2ColConvolution = 49,
    HwFcRelayout = 56,
    Clamp = 57,
    RefConvolution = 58,
    GlobalAvgPool = 59,
    GlobalMaxPool = 60,
    GRN = 61,
    MVN = 62,
    DepthDeconv = 63,
    Proposal = 64,
    ROIPooling = 65,
    PSROIPooling = 66,
    Interp = 67,
    Custom = 68,
    MTCNN = 69,
    LSTMCell = 70,
    Pad = 71,
    Resample = 72,
    Upsampling = 73,
    ArgMax = 74,
    Div = 75,
    Min = 76,
    Squared_diff = 77,
    Equal = 78,
    Not_equal = 79,
    Greater = 80,
    Greater_equal = 81,
    Less = 82,
    Less_equal = 83,
    Logical_NOT = 84,
    Logical_AND = 85,
    Logical_OR = 86,
    Logical_XOR = 87,
    Pow = 88,
    Floor_mod = 89,
    Select = 90,
    GEMM = 91,
    Log = 92,
    ReduceAnd = 93,
    ReverseSequence = 94,
    Gather = 100,
    Exp = 101,
    Floor = 102,
    TopK = 104,
    ReduceMin = 105,
)

//
// StageCategory
//

VPU_DECLARE_ENUM(StageCategory,
    SHAVE,
    HW,
    DMA,
    Special)

//
// PadMode
//

// Must be aligned with ie::PadLayer::ePadMode
VPU_DECLARE_ENUM(PadMode,
    Constant = 0,
    Edge = 1,
    Reflect = 2,
    Symmetric = 3)

//
// StageSHAVEsRequirements
//

VPU_DECLARE_ENUM(StageSHAVEsRequirements,
    NotNeeded,
    OnlyOne,
    TwoOrOne,
    CanBeLimited,
    NeedMax
);

//
// ScalePropagationStep
//

VPU_DECLARE_ENUM(ScalePropagationStep,
    Check,
    ScaleInput,
    Propagate
);

//
// TopKMode
//

// Firmware implementations must be aligned with these values
VPU_DECLARE_ENUM(TopKMode,
    Max = 0,
    Min = 1)

//
// TopKSort
//

// Firmware implementations must be aligned with these values
VPU_DECLARE_ENUM(TopKSort,
    None = 0,
    Value = 1,
    Index = 2)

//
// StageDataInfo
//

template <typename Val>
class StageDataInfo final {
public:
    StageDataInfo(const StageDataInfo&) = delete;
    StageDataInfo& operator=(const StageDataInfo&) = delete;

    StageDataInfo(StageDataInfo&&) = delete;
    StageDataInfo& operator=(StageDataInfo&&) = delete;

    inline void init(int numInputs, int numOutputs) {
        _inputVals.clear();
        _inputVals.resize(numInputs);

        _outputVals.clear();
        _outputVals.resize(numOutputs);
    }

    template <typename V>
    inline void setInput(const StageInput& edge, V&& val) {
        IE_ASSERT(edge->consumer().get() == _owner);
        IE_ASSERT(edge->portInd() >= 0 && edge->portInd() < _inputVals.size());
        _inputVals[edge->portInd()] = std::forward<V>(val);
    }
    template <typename V>
    inline void setOutput(const StageOutput& edge, V&& val) {
        IE_ASSERT(edge->producer().get() == _owner);
        IE_ASSERT(edge->portInd() >= 0 && edge->portInd() < _outputVals.size());
        _outputVals[edge->portInd()] = std::forward<V>(val);
    }

    inline bool hasInput(const StageInput& edge) const {
        IE_ASSERT(edge->consumer().get() == _owner);
        IE_ASSERT(edge->portInd() >= 0 && edge->portInd() < _inputVals.size());
        return _inputVals[edge->portInd()].hasValue();
    }
    inline bool hasOutput(const StageOutput& edge) const {
        IE_ASSERT(edge->producer().get() == _owner);
        IE_ASSERT(edge->portInd() >= 0 && edge->portInd() < _outputVals.size());
        return _outputVals[edge->portInd()].hasValue();
    }

    inline const Val& getInput(const StageInput& edge) const {
        IE_ASSERT(edge->consumer().get() == _owner);
        IE_ASSERT(edge->portInd() >= 0 && edge->portInd() < _inputVals.size());
        return _inputVals[edge->portInd()].get();
    }
    inline const Val& getOutput(const StageOutput& edge) const {
        IE_ASSERT(edge->producer().get() == _owner);
        IE_ASSERT(edge->portInd() >= 0 && edge->portInd() < _outputVals.size());
        return _outputVals[edge->portInd()].get();
    }

    bool empty() const {
        for (const auto& val : _inputVals) {
            if (val.hasValue()) {
                return false;
            }
        }
        for (const auto& val : _outputVals) {
            if (val.hasValue()) {
                return false;
            }
        }
        return true;
    }

private:
    friend class StageNode;
    explicit StageDataInfo(const StageNode* owner) : _owner(owner) {}

private:
    const StageNode* _owner = nullptr;

    SmallVector<Optional<Val>> _inputVals;
    SmallVector<Optional<Val>> _outputVals;
};

//
// StageNode
//

class StageNode :
        public EnableHandle,
        public EnableCustomAttributes,
        public std::enable_shared_from_this<StageNode> {
    //
    // Main attributes
    //

    VPU_MODEL_ATTRIBUTE(std::string, name, "")
    VPU_MODEL_ATTRIBUTE(StageType, type, StageType::None)
    VPU_MODEL_ATTRIBUTE(int, index, -1)

    //
    // Bindings with IE
    //

    VPU_MODEL_ATTRIBUTE(ie::CNNLayerPtr, origLayer, nullptr)

    //
    // Edges
    //

    VPU_MODEL_ATTRIBUTE_PTR_RANGE(StageInputVector, inputEdges)
    VPU_MODEL_ATTRIBUTE_PTR_RANGE(StageOutputVector, outputEdges)

    VPU_MODEL_ATTRIBUTE_PTR_RANGE(StageTempBufferVector, tempBufferEdges)

    VPU_MODEL_ATTRIBUTE(InjectedStage, parentStageEdge, nullptr)
    VPU_MODEL_ATTRIBUTE_PTR_RANGE(InjectedStageList, injectedStageEdges)

    //
    // SHAVEs allocation
    //

    VPU_MODEL_ATTRIBUTE(int, numSHAVEs, 0)

    VPU_MODEL_ATTRIBUTE(Handle<Model>, model, nullptr)

public:
    //
    // Comparison operators
    //

    struct NameCmp final {
        inline bool operator()(const Stage& left, const Stage& right) const {
            auto res = left->name().compare(right->name());
            if (res == 0) {
               // Different Stage nodes might have equal names, compare pointers instead to distinguish them.
               return reinterpret_cast<uintptr_t>(left.get()) < reinterpret_cast<uintptr_t>(right.get());
            }
            return res < 0;
        }
    };

    using NameOrderedSet = std::set<Stage, NameCmp>;

private:
    //
    // Range helpers
    //

    struct InputAccess final {
        inline auto operator()(const StageInput& edge) const -> decltype(edge->input()) {
            return edge->input();
        }
    };

    struct OutputAccess final {
        inline auto operator()(const StageOutput& edge) const -> decltype(edge->output()) {
            return edge->output();
        }
    };

    struct ProducerAccess final {
        inline auto operator()(const Data& data) const -> decltype(data->producer()) {
            return data->producer();
        }
    };

    struct ConsumersAccess final {
        inline auto operator()(const Data& data) const -> decltype(data->consumers()) {
            return data->consumers();
        }
    };

    struct TempBufferAccess final {
        inline auto operator()(const StageTempBuffer& edge) const -> decltype(edge->tempBuffer()) {
            return edge->tempBuffer();
        }
    };

    struct InjectedStageAccess final {
        inline auto operator()(const InjectedStage& edge) const -> decltype(edge->child()) {
            return edge->child();
        }
    };

    // It holds the number of separate edges per each prev/next stage
    using StageOrderMap = std::map<Stage, int, NameCmp>;

    struct StageOrderMapAccess final {
        inline const Stage& operator()(const StageOrderMap::value_type& p) const {
            return p.first;
        }
    };

    StageOrderMap _prevStages;
    StageOrderMap _nextStages;

public:
    inline int numInputs() const { return _inputEdges.size(); }
    inline StageInput inputEdge(int ind) const {
        IE_ASSERT(ind >= 0 && ind < _inputEdges.size());
        return _inputEdges[ind];
    }
    inline Data input(int ind) const {
        IE_ASSERT(ind >= 0 && ind < _inputEdges.size());
        return _inputEdges[ind]->input();
    }
    inline auto inputs() const -> decltype(mapRange<InputAccess>(inputEdges())) {
        return mapRange<InputAccess>(inputEdges());
    }

    inline int numOutputs() const { return _outputEdges.size(); }
    inline StageOutput outputEdge(int ind) const {
        IE_ASSERT(ind >= 0 && ind < _outputEdges.size());
        return _outputEdges[ind];
    }
    inline Data output(int ind) const {
        IE_ASSERT(ind >= 0 && ind < _outputEdges.size());
        return _outputEdges[ind]->output();
    }
    inline auto outputs() const -> decltype(mapRange<OutputAccess>(outputEdges())) {
        return mapRange<OutputAccess>(outputEdges());
    }

    inline auto prevStages() const -> decltype(_prevStages | asRange() | map<StageOrderMapAccess>()) {
        return _prevStages | asRange() | map<StageOrderMapAccess>();
    }
    inline auto nextStages() const -> decltype(_nextStages | asRange() | map<StageOrderMapAccess>()) {
        return _nextStages | asRange() | map<StageOrderMapAccess>();
    }

    inline int numTempBuffers() const { return _tempBufferEdges.size(); }
    inline StageTempBuffer tempBufferEdge(int ind) const {
        IE_ASSERT(ind >= 0 && ind < _tempBufferEdges.size());
        return _tempBufferEdges[ind];
    }
    inline Data tempBuffer(int ind) const {
        IE_ASSERT(ind >= 0 && ind < _tempBufferEdges.size());
        return _tempBufferEdges[ind]->tempBuffer();
    }
    inline auto tempBuffers() const -> decltype(mapRange<TempBufferAccess>(tempBufferEdges())) {
        return mapRange<TempBufferAccess>(tempBufferEdges());
    }

    inline Stage parentStage() const { return _parentStageEdge == nullptr ? nullptr : _parentStageEdge->parent(); }

    inline int numInjectedStages() const {
        return _injectedStageEdges.size();
    }
    inline auto injectedStages() const -> decltype(mapRange<InjectedStageAccess>(injectedStageEdges())) {
        return mapRange<InjectedStageAccess>(injectedStageEdges());
    }

public:
    inline virtual ~StageNode() = default;

    //
    // Stage category
    //

    inline StageCategory category() const {
        if (static_cast<int>(_type) < 0) {
            return StageCategory::Special;
        } else if (_type == StageType::MyriadXHwOp) {
            return StageCategory::HW;
        } else if (_type == StageType::Copy) {
            return StageCategory::DMA;
        } else {
            return StageCategory::SHAVE;
        }
    }

    //
    // Bindings with IE
    //

    inline std::string origLayerName() const {
        return _origLayer != nullptr ? _origLayer->name : std::string();
    }

    //
    // SHAVEs allocation
    //

    void setNumSHAVEs(int numSHAVEs);

    //
    // Passes utilities
    //

    // Scale factor propagation from inputs to outputs
    const StageDataInfo<float>& propagateScaleFactors(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step);

    // Data order propagation from inputs to outputs.
    const StageDataInfo<DimsOrder>& propagateDataOrder();

    // Get Data strides requirements
    const StageDataInfo<StridesRequirement>& getDataStridesRequirements();

    // Finalize internal parameter to final Data layout.
    void finalizeDataLayout();

    // Information about batch support.
    const StageDataInfo<BatchSupport>& getBatchSupportInfo();

    // Resources requirements.
    StageSHAVEsRequirements getSHAVEsRequirements() const;

    void initialCheck() const;
    void finalCheck() const;

    // Name postfix for modified stage
    inline void appendNamePostfix(const std::string& postfix) {
        _name = _name + postfix;
    }

    StageListNode posForPassList;

    //
    // Backend utilities
    //

    void serialize(BlobSerializer& serializer) const;

protected:
    //
    // Interfaces for Stages implementations
    //

    virtual StagePtr cloneImpl() const = 0;

    virtual void propagateScaleFactorsImpl(
            const SmallVector<float>& inputScales,
            ScalePropagationStep step,
            StageDataInfo<float>& scaleInfo);

    virtual void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) = 0;

    virtual void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) = 0;

    virtual void finalizeDataLayoutImpl() = 0;

    virtual void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) = 0;

    virtual StageSHAVEsRequirements getSHAVEsRequirementsImpl() const;

    virtual void initialCheckImpl() const {}
    virtual void finalCheckImpl() const {}

    virtual void serializeParamsImpl(BlobSerializer& serializer) const = 0;

    virtual void serializeDataImpl(BlobSerializer& serializer) const = 0;

protected:
    inline StageNode() :
            posForPassList(this),
            _injectedStageEdges(&InjectedStageEdge::_posInStage),
            _scaleInfo(this),
            _orderInfo(this),
            _stridesInfo(this),
            _batchInfo(this),
            _posInModel(this) {
    }
    inline StageNode(const StageNode& other) :
            EnableCustomAttributes(other),
            posForPassList(this),
            _injectedStageEdges(&InjectedStageEdge::_posInStage),
            _scaleInfo(this),
            _orderInfo(this),
            _stridesInfo(this),
            _batchInfo(this),
            _posInModel(this) {
    }

    void changeType(StageType type) {
        _type = type;
    }

private:
    StageDataInfo<float> _scaleInfo;
    StageDataInfo<DimsOrder> _orderInfo;
    StageDataInfo<StridesRequirement> _stridesInfo;
    StageDataInfo<BatchSupport> _batchInfo;

    StagePtrList::iterator _ptrPosInModel;
    StageListNode _posInModel;

    friend class Model;
};

void printTo(std::ostream& os, const Stage& stage);

void assertAllInputsOutputsTypes(const Stage& stage,
                                 const DataType& expectedInputsType,
                                 const DataType& expectedOutputsType);

void assertInputsOutputsTypes(const Stage& stage,
                              const std::vector<EnumSet<DataType>>& expectedInputsTypes,
                              const std::vector<EnumSet<DataType>>& expectedOutputsTypes);

}  // namespace vpu
