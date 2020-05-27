// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/base.hpp>

namespace vpu {

//
// Data -> Stage edges.
//

//
// StageInputEdge
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
    Model _model;
    StageInputPtrList::iterator _ptrPosInModel;
    StageInputListNode _posInData;

    friend ModelObj;
    friend DataNode;
};

//
// StageDependencyEdge defines that some data should be calculated before the stage starts
// but this data is not an input for the stage, e.g. this data is used as a shape for stage output.
//

class StageDependencyEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
VPU_MODEL_ATTRIBUTE(Data, dependency, nullptr)
VPU_MODEL_ATTRIBUTE(Stage, dependentStage, nullptr)

private:
    StageDependencyEdge() : _posInData(this) {}

private:
    StageDependencyPtrList::iterator _ptrPosInModel;
    StageDependencyListNode _posInData;

    friend ModelObj;
    friend DataNode;
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
    Model _model;
    StageOutputPtrList::iterator _ptrPosInModel;

    friend ModelObj;
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
    Model _model;
    StageTempBufferPtrList::iterator _ptrPosInModel;

    friend ModelObj;
};

//
// DataToDataAllocationEdge
//

//
// Data <-> Data edges - used to share memory buffer between Data objects.
// Parent Data object owns the memory, while child reuses it.
//

//
// SharedDataMode defines the relationship between the Data objects:
//    * ROI : child is a sub-tensor of parent.
//      They have the same layout and strides, but child has smaller dimensions.
//    * Reshape : used for Reshape operation.
//      Child shares the same memory buffer, but has completely different layout.
//

VPU_DECLARE_ENUM(SharedDataMode,
    ROI,
    Reshape)

//
// SharedDataOrder defined the Data flow order between parent and child.
//    * ParentWritesToChild :
//      (Producer) -> [Parent] -> [Child] -> (Consumer)
//    * ChildWritesToParent :
//      (Producer) -> [Child] -> [Parent] -> (Consumer)
//

VPU_DECLARE_ENUM(SharedDataOrder,
    ParentWritesToChild,
    ChildWritesToParent)

//
// SharedConnectionMode defines if there should be connection stage between Data objects:
//    * SINGLE_STAGE : there must exactly one connection stage.
//    * SUBGRAPH : there may be more than one stage between data objects
//      so we cannot define any of them as a connection stage. It is intended to disable
//      checks for connection stage type.
//

VPU_DECLARE_ENUM(SharedConnectionMode,
    SINGLE_STAGE,
    SUBGRAPH)

class DataToDataAllocationEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Data, parent, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, child, nullptr)
    VPU_MODEL_ATTRIBUTE(Stage, connection, nullptr)
    VPU_MODEL_ATTRIBUTE(SharedDataMode, mode, SharedDataMode::ROI)
    VPU_MODEL_ATTRIBUTE(SharedDataOrder, order, SharedDataOrder::ParentWritesToChild)
    VPU_MODEL_ATTRIBUTE(SharedConnectionMode, connectionMode, SharedConnectionMode::SINGLE_STAGE);

private:
    DataToDataAllocationEdge() : _posInData(this) {}

private:
    Model _model;
    DataToDataAllocationPtrList::iterator _ptrPosInModel;
    DataToDataAllocationListNode _posInData;

    friend ModelObj;
    friend DataNode;
};

//
// DataToShapeAllocationEdge
//

//
// Data <-> Shape of data edge - used to share data memory of one DataNode as shape for another DataNode
//

class DataToShapeAllocationEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Data, parent, nullptr)
    VPU_MODEL_ATTRIBUTE(Data, child, nullptr)

private:
    DataToShapeAllocationEdge() : _posInData(this) {}

private:
    DataToShapeAllocationPtrList::iterator _ptrPosInModel;
    DataToShapeAllocationListNode _posInData;

    friend ModelObj;
    friend DataNode;
};

//
// InjectionEdge
//

//
// Stage <-> Stage edges - used to inject SW operations into HW
//

class InjectionEdge final :
        public EnableHandle,
        public EnableCustomAttributes {
    VPU_MODEL_ATTRIBUTE(Stage, parent, nullptr)
    VPU_MODEL_ATTRIBUTE(StagePtr, child, nullptr)
    VPU_MODEL_ATTRIBUTE(int, portInd, -1)

private:
    InjectionEdge() : _posInStage(this) {}

private:
    Model _model;
    InjectionPtrList::iterator _ptrPosInModel;
    InjectionListNode _posInStage;

    friend ModelObj;
    friend StageNode;
};

}  // namespace vpu
