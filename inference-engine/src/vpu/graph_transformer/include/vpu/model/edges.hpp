// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/base.hpp>

namespace vpu {

//
// StageInputEdge
//

//
// Data -> Stage edge.
//

class StageInputEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Data, input, nullptr)
    VPU_MODEL_ATTRIBUTE(Stage, consumer, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)
    VPU_MODEL_ATTRIBUTE(StageInput, parentEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(StageInput, childEdge, nullptr)

private:
    StageInputEdge() : _posInData(this) {}

private:
    Handle<Model> _model;
    StageInputPtrList::iterator _ptrPosInModel;
    StageInputListNode _posInData;

    friend class Model;
    friend class DataNode;
};

//
// StageOutputEdge
//

//
// Stage -> Data edge.
//

class StageOutputEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Stage, producer, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, output, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)
    VPU_MODEL_ATTRIBUTE(StageOutput, parentEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(StageOutput, childEdge, nullptr)

private:
    Handle<Model> _model;
    StageOutputPtrList::iterator _ptrPosInModel;

    friend class Model;
};

//
// StageTempBufferEdge
//

class StageTempBufferEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Stage, stage, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, tempBuffer, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)
    VPU_MODEL_ATTRIBUTE(StageTempBuffer, parentEdge, nullptr)
    VPU_MODEL_ATTRIBUTE(StageTempBuffer, childEdge, nullptr)

private:
    Handle<Model> _model;
    StageTempBufferPtrList::iterator _ptrPosInModel;

    friend class Model;
};

//
// SharedAllocationEdge
//

//
// Data <-> Data edges - used to share memory buffer between Data objects.
// Parent Data object owns the memory, while child reuses it.
//
// SharedDataMode defines the relationship between the Data objects:
//    * ROI : child is a sub-tensor of parent.
//      They have the same layout and strides, but child has smaller dimensions.
//    * Reshape : used for Reshape operation.
//      Child shares the same memory buffer, but has completely different layout.
//
// SharedDataOrder defined the Data flow order between parent and child.
//    * ParentWritesToChild :
//      (Producer) -> [Parent] -> [Child] -> (Consumer)
//    * ChildWritesToParent :
//      (Producer) -> [Child] -> [Parent] -> (Consumer)
//

VPU_DECLARE_ENUM(SharedDataMode,
    ROI,
    Reshape)

VPU_DECLARE_ENUM(SharedDataOrder,
    ParentWritesToChild,
    ChildWritesToParent)

class SharedAllocationEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Data, parent, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, child, nullptr)
    VPU_MODEL_ATTRIBUTE(Stage, connection, nullptr)
    VPU_MODEL_ATTRIBUTE(SharedDataMode, mode, SharedDataMode::ROI)
    VPU_MODEL_ATTRIBUTE(SharedDataOrder, order, SharedDataOrder::ParentWritesToChild)

private:
    SharedAllocationEdge() : _posInData(this) {}

private:
    Handle<Model> _model;
    SharedAllocationPtrList::iterator _ptrPosInModel;
    SharedAllocationListNode _posInData;

    friend class Model;
    friend class DataNode;
};

//
// InjectedStageEdge
//

//
// Stage <-> Stage edges - used to inject SW operations into HW
//

class InjectedStageEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Stage, parent, nullptr)
    VPU_MODEL_ATTRIBUTE(StagePtr, child, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)

private:
    InjectedStageEdge() : _posInStage(this) {}

private:
    Handle<Model> _model;
    InjectedStagePtrList::iterator _ptrPosInModel;
    InjectedStageListNode _posInStage;

    friend class Model;
    friend class StageNode;
};

}  // namespace vpu
