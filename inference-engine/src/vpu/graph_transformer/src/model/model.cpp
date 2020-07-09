// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/model.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <details/caseless.hpp>
#include "blob_factory.hpp"

#include <cctype>
#include <memory>
#include <string>
#include <set>
#include <exception>
#include <algorithm>

namespace vpu {

//
// Resources
//

void printTo(std::ostream& os, const Resources& res) {
    os << "[" << std::endl;

    os << "numCMXSlices=" << res.numCMXSlices << std::endl;
    os << "numSHAVEs=" << res.numSHAVEs << std::endl;

    os << "]";
}

void printTo(DotLabel& lbl, const Resources& res) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("numCMXSlices", res.numCMXSlices);
    subLbl.appendPair("numSHAVEs", res.numSHAVEs);
}

//
// Model
//

void ModelObj::setBatchSize(int batchSize) {
    // Check `batchSize` value.
    VPU_THROW_UNLESS(
        batchSize >= 1,
        "Unexpected network batch size : %v", batchSize);

    _batchSize = batchSize;
    _allocator.setBatchSize(batchSize);
}

Data ModelObj::addInputData(
        const std::string& name,
        const DataDesc& desc) {
    std::shared_ptr<DataNode> data(new DataNode);

    data->_name = name;
    data->_usage = DataUsage::Input;
    data->_desc = desc;
    data->_model = this;

    data->_ptrPosInModel = _dataPtrList.emplace(_dataPtrList.end(), data);
    _dataList.push_back(data);

    _allocator.setNeedToAllocNonIntermData();

    return data;
}

Data ModelObj::addOutputData(
        const std::string& name,
        const DataDesc& desc) {
    std::shared_ptr<DataNode> data(new DataNode);

    data->_name = name;
    data->_usage = DataUsage::Output;
    data->_desc = desc;
    data->_model = this;

    data->_ptrPosInModel = _dataPtrList.emplace(_dataPtrList.end(), data);
    _dataList.push_back(data);

    _allocator.setNeedToAllocNonIntermData();

    return data;
}

Data ModelObj::addConstData(
        const std::string& name,
        const DataDesc& desc,
        const DataContent::Ptr& content) {
    IE_ASSERT(content != nullptr);

    VPU_THROW_UNLESS(desc.totalDimSize() * desc.elemSize() == content->byteSize(),
        "addConstData error: while duplicating {} Const data got different "
        "newDesc and content byte sizes ({} and {} respectively)",
        name, desc.totalDimSize() * desc.elemSize(), content->byteSize());

    std::shared_ptr<DataNode> data(new DataNode);

    data->_name = name;
    data->_usage = DataUsage::Const;
    data->_desc = desc;
    data->_model = this;

    data->_content = content;

    data->_ptrPosInModel = _dataPtrList.emplace(_dataPtrList.end(), data);
    _dataList.push_back(data);

    _allocator.setNeedToAllocNonIntermData();

    return data;
}

Data ModelObj::addConstData(const std::string& name, const DataDesc& descriptor, const std::function<void(const ie::Blob::Ptr&)>& generator) {
    const auto ieBlob = make_blob_with_precision(descriptor.toTensorDesc());
    ieBlob->allocate();
    if (generator) {
        generator(ieBlob);
    }
    return addConstData(name, descriptor, ieBlobContent(ieBlob, descriptor.type()));
}

Data ModelObj::addNewData(
        const std::string& name,
        const DataDesc& desc) {
    std::shared_ptr<DataNode> data(new DataNode);

    data->_name = name;
    data->_usage = DataUsage::Intermediate;
    data->_desc = desc;
    data->_model = this;

    data->_ptrPosInModel = _dataPtrList.emplace(_dataPtrList.end(), data);
    _dataList.push_back(data);

    return data;
}

Data ModelObj::addFakeData() {
    std::shared_ptr<DataNode> data(new DataNode);

    data->_name = "<fake>";
    data->_usage = DataUsage::Fake;
    data->_desc = DataDesc({1});
    data->_model = this;

    data->_ptrPosInModel = _dataPtrList.emplace(_dataPtrList.end(), data);
    _dataList.push_back(data);

    return data;
}

Data ModelObj::duplicateData(
        const Data& origData,
        const std::string& postfix,
        const DataDesc& newDesc,
        const DataContent::Ptr& newContent) {
    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(origData->_model.get() == this);

    //
    // Duplicate Data node.
    //

    auto newDataUsage = origData->usage();
    if (newDataUsage == DataUsage::Input ||
        newDataUsage == DataUsage::Output) {
        // Duplicates for Input & Output can be only Intermediate
        newDataUsage = DataUsage::Intermediate;
    }

    std::shared_ptr<DataNode> newData(new DataNode);

    newData->_name  = origData->name() + postfix;
    newData->_usage = newDataUsage;
    newData->_desc  = newDesc.numDims() != 0 ? newDesc : origData->desc();
    newData->_model = this;

    if (const auto& parentDataToShapeEdge = origData->parentDataToShapeEdge()) {
        connectDataWithShape(parentDataToShapeEdge->parent(), newData);
    }

    if (newDataUsage == DataUsage::Const) {
        const auto& content = newContent != nullptr ? newContent : origData->content();
        const auto& desc = newDesc != DataDesc() ? newDesc : origData->desc();

        VPU_THROW_UNLESS(desc.totalDimSize() * desc.elemSize() == content->byteSize(),
            "duplicateData error: while duplicating {} Const data got different "
            "desc and content byte sizes ({} and {} respectively)",
            origData->name(), desc.totalDimSize() * desc.elemSize(), content->byteSize());

        newData->_content = content;
    }

    newData->attrs().copyFrom(origData->attrs());

    newData->_ptrPosInModel = _dataPtrList.emplace(_dataPtrList.end(), newData);
    _dataList.push_back(newData);

    return newData;
}

Stage ModelObj::duplicateStage(
        const Stage& origStage,
        const std::string& postfix,
        const DataVector& inputs,
        const DataVector& outputs) {
    //
    // Check that the new Stage has inputs and outputs.
    //

    IE_ASSERT(!inputs.empty());
    IE_ASSERT(!outputs.empty());

    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(origStage->_model.get() == this);

    for (const auto& input : inputs) {
        IE_ASSERT(input->_model.get() == this);
    }

    for (const auto& output : outputs) {
        IE_ASSERT(output->_model.get() == this);
    }

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            IE_ASSERT(input != output);
        }
    }

    //
    // Create new Stage.
    //

    _resetStageOrder = true;

    auto stage = origStage->cloneImpl();

    stage->_name      = origStage->name() + postfix;
    stage->_id        = _stagesIdCount++;
    stage->_type      = origStage->_type;
    stage->_origLayer = origStage->_origLayer;
    stage->_model     = this;

    _initialStages.emplace(stage);

    for (const auto& input : inputs) {
        addStageInput(stage, input);
    }
    for (const auto& output : outputs) {
        addStageOutput(stage, output);
    }
    for (const auto& tempBufferEdge : origStage->_tempBufferEdges) {
        addTempBuffer(stage, tempBufferEdge->tempBuffer()->desc());
    }

    stage->_ptrPosInModel = _stagePtrList.emplace(_stagePtrList.end(), stage);

    return stage;
}

StageInput ModelObj::addStageInput(
        const Stage& stage,
        const Data& data) {
    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);
    IE_ASSERT(data->_model.get() == this);

    // TODO: check for loops in the graph.

    //
    // Input data can't be Temp.
    //

    IE_ASSERT(data->_usage != DataUsage::Temp);

    //
    // Create new Edge.
    //

    _resetStageOrder = true;

    std::shared_ptr<StageInputEdge> edge(new StageInputEdge);

    edge->_consumer = stage;
    edge->_input = data;
    edge->_portInd = stage->_inputEdges.size();
    edge->_model = this;

    edge->_ptrPosInModel = _inEdgePtrList.emplace(_inEdgePtrList.end(), edge);
    data->_consumerEdges.push_back(edge);
    stage->_inputEdges.emplace_back(edge);

    //
    // Stage order helpers
    //

    if (data->_producerEdge != nullptr) {
        IE_ASSERT(stage->_parentStageEdge == nullptr);
        IE_ASSERT(data->_producerEdge->_producer->_parentStageEdge == nullptr);
        setStagesOrder(data->producerEdge()->producer(), stage);
    }

    if (stage->_prevStages.empty()) {
        _initialStages.emplace(stage);
    } else {
        _initialStages.erase(stage);
    }

    return edge;
}

StageOutput ModelObj::addStageOutput(
        const Stage& stage,
        const Data& data) {
    //
    // Check that the objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);
    IE_ASSERT(data->_model.get() == this);

    //
    // Check that the `data` is free.
    //

    IE_ASSERT(data->_producerEdge == nullptr);

    if (data->_parentDataToDataEdge != nullptr) {
        IE_ASSERT(data->_parentDataToDataEdge->_order != SharedDataOrder::ParentWritesToChild);
    }

    for (const auto& childDataEdge : data->_childDataToDataEdges) {
        IE_ASSERT(childDataEdge->_order != SharedDataOrder::ChildWritesToParent);
    }

    //
    // Output data can be Output, Intermediate, or Fake only.
    //

    IE_ASSERT(data->_usage == DataUsage::Output || data->_usage == DataUsage::Intermediate || data->_usage == DataUsage::Fake);

    // TODO: check for loops in the graph.

    _resetStageOrder = true;

    std::shared_ptr<StageOutputEdge> edge(new StageOutputEdge);

    edge->_producer = stage;
    edge->_output = data;
    edge->_portInd = stage->_outputEdges.size();
    edge->_model = this;

    edge->_ptrPosInModel = _outEdgePtrList.emplace(_outEdgePtrList.end(), edge);
    stage->_outputEdges.emplace_back(edge);
    data->_producerEdge = edge;

    //
    // Stage order helpers
    //

    for (const auto& consumerEdge : data->_consumerEdges) {
        IE_ASSERT(stage->_parentStageEdge == nullptr);
        IE_ASSERT(consumerEdge->_consumer->_parentStageEdge == nullptr);
        setStagesOrder(stage, consumerEdge->consumer());
    }

    return edge;
}

StageDependency ModelObj::addStageDependency(const Stage& stage, const Data& data) {
    for (const auto& dependentStageEdge : data->dependentStagesEdges()) {
        VPU_THROW_UNLESS(dependentStageEdge->dependentStage() != stage,
                         "Adding stage dependency for {} with type {} failed: data {} with usage {} is already its dependency",
                         stage->name(), stage->type(), data->name(), data->usage());
    }

    for (const auto& input : stage->inputs()) {
        VPU_THROW_UNLESS(data != input,
                         "Adding stage dependency for {} with type {} failed: data {} with usage {} is already its input",
                         stage->name(), stage->type(), data->name(), data->usage());
    }

    VPU_THROW_UNLESS(data->producer() != nullptr,
                     "Adding stage dependency for {} with type {} failed: data {} with usage {} should have producer, "
                     "but actually it doesn't", stage->name(), stage->type(), data->name(), data->usage());

    _resetStageOrder = true;

    std::shared_ptr<StageDependencyEdge> edge(new StageDependencyEdge);
    edge->_ptrPosInModel = _stageDependencyEdgePtrList.emplace(_stageDependencyEdgePtrList.end(), edge);

    edge->_dependency = data;
    edge->_dependentStage = stage;

    data->_dependentStagesEdges.push_back(edge);

    setStagesOrder(data->producerEdge()->producer(), stage);

    return edge;
}

StageTempBuffer ModelObj::addTempBuffer(
        const Stage& stage,
        const DataDesc& desc) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);

    //
    // Create new Data.
    //

    std::shared_ptr<DataNode> data(new DataNode);

    data->_name = formatString("%s@temp@%d", stage->name(), stage->_tempBufferEdges.size() + 1);
    data->_usage = DataUsage::Temp;
    data->_desc = desc;
    data->_model = this;

    data->_ptrPosInModel = _dataPtrList.emplace(_dataPtrList.end(), data);
    _dataList.push_back(data);

    //
    // Create new Edge.
    //

    std::shared_ptr<StageTempBufferEdge> edge(new StageTempBufferEdge);

    edge->_stage = stage;
    edge->_tempBuffer = data;
    edge->_portInd = stage->_tempBufferEdges.size();
    edge->_model = this;

    edge->_ptrPosInModel = _tempBufferEdgePtrList.emplace(_tempBufferEdgePtrList.end(), edge);
    stage->_tempBufferEdges.emplace_back(edge);
    data->_tempBufferEdge = edge;

    return edge;
}

void ModelObj::replaceStageInput(
        const StageInput& edge,
        const Data& newInput) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(edge->_model.get() == this);
    IE_ASSERT(newInput->_model.get() == this);

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& output : edge->consumer()->outputs()) {
        IE_ASSERT(newInput != output);
    }

    //
    // Input data can't be Temp.
    //

    IE_ASSERT(newInput->_usage != DataUsage::Temp);

    //
    // Can't replace Edge from injected Stage.
    //

    IE_ASSERT(edge->_parentEdge == nullptr);
    IE_ASSERT(edge->_childEdge == nullptr);

    //
    // New and old dynamic data must have the same parent shape data
    //

    if (const auto& oldParentDataToShapeEdge = edge->input()->parentDataToShapeEdge()) {
        const auto& newParentDataToShapeEdge = newInput->parentDataToShapeEdge();
        VPU_THROW_UNLESS(newParentDataToShapeEdge != nullptr,
                "Replaced input data with name {} from {} stage with name {} has parentDataToShapeEdge, "
                "but new input data with name {} has no parentDataToShapeEdge",
                edge->input()->name(), edge->consumer()->type(), edge->consumer()->name(), newInput->name());
        VPU_THROW_UNLESS(newParentDataToShapeEdge->parent() == oldParentDataToShapeEdge->parent(),
                "Replaced input data with name {} from {} stage with name {} and new input data with name must "
                "have the same shape data",
                edge->input()->name(), edge->consumer()->type(), edge->consumer()->name(), newInput->name());
    } else {
        VPU_THROW_UNLESS(newInput->parentDataToShapeEdge() == nullptr,
                "Replaced input data with name {} from {} stage with name {} has not parentDataToShapeEdge, "
                "but new input data with name {} has",
                edge->input()->name(), edge->consumer()->type(), edge->consumer()->name(), newInput->name());
    }

    //
    // Edge change affects the Stage order.
    //

    _resetStageOrder = true;

    //
    // Remove Edge from previous input.
    //

    edge->_input->_consumerEdges.erase(edge);

    //
    // Previous stage order helpers
    //

    if (edge->_input->_producerEdge != nullptr) {
        removeStagesOrder(edge->input()->producer(), edge->consumer());
    }

    //
    // Set new input.
    //

    edge->_input = newInput;
    newInput->_consumerEdges.push_back(edge);

    //
    // Stage order helpers
    //

    if (newInput->_producerEdge != nullptr) {
        IE_ASSERT(edge->_consumer->_parentStageEdge == nullptr);
        IE_ASSERT(newInput->_producerEdge->_producer->_parentStageEdge == nullptr);
        setStagesOrder(newInput->producerEdge()->producer(), edge->consumer());
    }

    if (edge->_consumer->_prevStages.empty()) {
        _initialStages.emplace(edge->_consumer);
    } else {
        _initialStages.erase(edge->_consumer);
    }
}

void ModelObj::replaceStageOutput(
        const StageOutput& edge,
        const Data& newOutput) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(edge->_model.get() == this);
    IE_ASSERT(newOutput->_model.get() == this);

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& input : edge->producer()->inputs()) {
        IE_ASSERT(newOutput != input);
    }

    //
    // Check that `data` is free.
    //

    IE_ASSERT(newOutput->_producerEdge == nullptr);

    if (newOutput->_parentDataToDataEdge != nullptr) {
        IE_ASSERT(newOutput->_parentDataToDataEdge->_order != SharedDataOrder::ParentWritesToChild);
    }

    for (const auto& childDataEdge : newOutput->_childDataToDataEdges) {
        IE_ASSERT(childDataEdge->_order != SharedDataOrder::ChildWritesToParent);
    }

    //
    // Output data can be Output/Intermediate/Fake.
    //

    IE_ASSERT(newOutput->_usage == DataUsage::Output ||
              newOutput->_usage == DataUsage::Intermediate ||
              newOutput->_usage == DataUsage::Fake);

    //
    // Can't replace Edge from injected Stage.
    //

    IE_ASSERT(edge->_parentEdge == nullptr);
    IE_ASSERT(edge->_childEdge == nullptr);

    //
    // New and old dynamic data must have the same parent shape data
    //

    if (const auto& oldParentDataToShapeEdge = edge->output()->parentDataToShapeEdge()) {
        const auto& newParentDataToShapeEdge = newOutput->parentDataToShapeEdge();
        VPU_THROW_UNLESS(newParentDataToShapeEdge != nullptr,
                "Replaced output data with name {} from {} stage with name {} has parentDataToShapeEdge, "
                "but new output data with name {} has no parentDataToShapeEdge",
                edge->output()->name(), edge->producer()->type(), edge->producer()->name(), newOutput->name());
        VPU_THROW_UNLESS(newParentDataToShapeEdge->parent() == oldParentDataToShapeEdge->parent(),
                "Replaced output data with name {} from {} stage with name {} and new output data with name must "
                "have the same shape data",
                edge->output()->name(), edge->producer()->type(), edge->producer()->name(), newOutput->name());
    } else {
        VPU_THROW_UNLESS(newOutput->parentDataToShapeEdge() == nullptr,
                "Replaced output data with name {} from {} stage with name {} has not parentDataToShapeEdge, "
                "but new output data with name {} has",
                edge->output()->name(), edge->producer()->type(), edge->producer()->name(), newOutput->name());
    }

    //
    // Edge change affects the Stage order.
    //

    _resetStageOrder = true;

    //
    // Remove Edge from previous output.
    //

    edge->_output->_producerEdge = nullptr;

    //
    // Previous stage order helpers
    //

    for (const auto& consumerEdge : edge->_output->_consumerEdges) {
        removeStagesOrder(edge->producer(), consumerEdge->consumer());

        if (consumerEdge->_consumer->_prevStages.empty()) {
            _initialStages.emplace(consumerEdge->_consumer);
        } else {
            _initialStages.erase(consumerEdge->_consumer);
        }
    }

    //
    // Set new output.
    //

    edge->_output = newOutput;
    newOutput->_producerEdge = edge;

    //
    // Stage order helpers
    //

    for (const auto& consumerEdge : newOutput->_consumerEdges) {
        IE_ASSERT(edge->_producer->_parentStageEdge == nullptr);
        IE_ASSERT(consumerEdge->_consumer->_parentStageEdge == nullptr);
        setStagesOrder(edge->producer(), consumerEdge->consumer());
    }
}

void ModelObj::replaceStageDependency(
        const StageDependency& edge,
        const Data& newDependency) {
    const auto previousDependency = edge->dependency();
    const auto dependentStage = edge->dependentStage();

    for (const auto& dependentStageEdge : newDependency->dependentStagesEdges()) {
        VPU_THROW_UNLESS(dependentStageEdge->dependentStage() != dependentStage,
            "replaceStageDependency failed for dependency {} with usage {} and dependentStage {} with type {}: "
            "new dependency {} with usage {} is already dependency for dependent stage", previousDependency->name(), previousDependency->usage(),
            dependentStage->name(), dependentStage->type(), newDependency->name(), newDependency->usage());
    }

    for (const auto& input : dependentStage->inputs()) {
        VPU_THROW_UNLESS(newDependency != input,
            "replaceStageDependency failed for dependency {} with usage {} and dependentStage {} with type {}: "
            "new dependency {} with usage {} is already input for dependent stage", previousDependency->name(), previousDependency->usage(),
            dependentStage->name(), dependentStage->type(), newDependency->name(), newDependency->usage());
    }

    VPU_THROW_UNLESS(newDependency->producer() != nullptr,
        "replaceStageDependency failed for dependency {} with usage {} and dependentStage {} with type {}: "
        "newDependency {} with usage {} has no producer", previousDependency->name(), previousDependency->usage(),
        dependentStage->name(), dependentStage->type(), newDependency->name(), newDependency->usage());

    VPU_THROW_UNLESS(previousDependency->producer() != nullptr,
        "replaceStageDependency failed for dependency {} with usage {} and dependentStage {} with type {}: "
        "previous dependency has no producer",
        previousDependency->name(), previousDependency->usage(), dependentStage->name(), dependentStage->type());

    _resetStageOrder = true;

    previousDependency->_dependentStagesEdges.erase(edge);

    removeStagesOrder(previousDependency->producer(), dependentStage);

    edge->_dependency = newDependency;
    newDependency->_dependentStagesEdges.push_back(edge);

    setStagesOrder(newDependency->producerEdge()->producer(), dependentStage);
}

void ModelObj::replaceDependentStage(
        const StageDependency& edge,
        const Stage& newDependentStage) {
    const auto dependency = edge->dependency();
    const auto previousDependentStage = edge->dependentStage();

    for (const auto& dependentStageEdge : dependency->dependentStagesEdges()) {
        VPU_THROW_UNLESS(dependentStageEdge->dependentStage() != newDependentStage,
            "replaceDependentStage failed for dependency {} with usage {} and dependentStage {} with type {}: "
            "new dependent stage {} with type {} is already dependent stage for dependency", dependency->name(), dependency->usage(),
            previousDependentStage->name(), previousDependentStage->type(), newDependentStage->name(), newDependentStage->type());
    }

    for (const auto& input : newDependentStage->inputs()) {
        VPU_THROW_UNLESS(dependency != input,
            "replaceDependentStage failed for dependency {} with usage {} and dependentStage {} with type {}: "
            "new dependent stage {} with type {} already has dependency as its input", dependency->name(), dependency->usage(),
            previousDependentStage->name(), previousDependentStage->type(), newDependentStage->name(), newDependentStage->type());
    }

    VPU_THROW_UNLESS(dependency->producer() != nullptr,
        "replaceDependentStage failed for dependency {} with usage {} and dependentStage {} with type {}: "
        "dependency has no producer",
        dependency->name(), dependency->usage(), previousDependentStage->name(), previousDependentStage->type());

    _resetStageOrder = true;

    removeStagesOrder(dependency->producer(), previousDependentStage);

    edge->_dependentStage = newDependentStage;

    setStagesOrder(dependency->producer(), newDependentStage);
}

void ModelObj::removeStageDependency(const StageDependency& edge) {
    const auto dependency = edge->dependency();
    const auto dependentStage = edge->dependentStage();

    VPU_THROW_UNLESS(dependency->producer(),
        "removeStageDependency failed for dependency {} with usage {} and dependentStage {} with type {}: dependency has no producer",
        dependency->name(), dependency->usage(), dependentStage->name(), dependentStage->type());

    _resetStageOrder = true;

    dependency->_dependentStagesEdges.erase(edge);

    removeStagesOrder(dependency->producer(), dependentStage);

    VPU_THROW_UNLESS(edge->_ptrPosInModel != _stageDependencyEdgePtrList.end(),
        "removeStageDependency failed for dependency {} with usage {} and dependentStage {} with type {}: no such edge in Model's DataToShapeEdges list",
        dependency->name(), dependency->usage(), dependentStage->name(), dependentStage->type());

    _stageDependencyEdgePtrList.erase(edge->_ptrPosInModel);
}

void ModelObj::removeStageDependency(const Stage& stage, const Data& dependency) {
    const auto& dependentStagesEdges = dependency->dependentStagesEdges();

    const auto it = std::find_if(dependentStagesEdges.begin(), dependentStagesEdges.end(), [&stage](const StageDependency& edge) {
        return edge->dependentStage() == stage;
    });

    if (it != dependentStagesEdges.end()) {
        const auto stageDependencyEdge = *it;
        removeStageDependency(stageDependencyEdge);
    }
}

ModelObj::InjectStageHelper::~InjectStageHelper() {
    //
    // Check that `done` was called.
    //

    if (_model != nullptr) {
        std::terminate();
    }
}

ModelObj::InjectStageHelper& ModelObj::InjectStageHelper::parentHW(const Stage& parent) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `parentHW` was not called.
    //

    IE_ASSERT(_parent == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(parent->_model == _model);

    //
    // Check that `parent` is HW.
    //

    IE_ASSERT(parent->category() == StageCategory::HW);

    _parent = parent;

    return *this;
}

ModelObj::InjectStageHelper& ModelObj::InjectStageHelper::childSW(const Stage& child) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `childSW` was not called.
    //

    IE_ASSERT(_child == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(child->_model == _model);

    //
    // Check that `parent` is HW.
    //

    IE_ASSERT(child->category() == StageCategory::DMA || child->category() == StageCategory::SHAVE);

    _child = child;

    return *this;
}

Injection ModelObj::InjectStageHelper::done() {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that all fields were set.
    //

    IE_ASSERT(_parent != nullptr);
    IE_ASSERT(_child != nullptr);

    //
    // Call actual implementation.
    //

    auto edge = _model->injectStageImpl(_parent, _child);

    //
    // Reset the internal state.
    //

    _model = nullptr;

    return edge;
}

Injection ModelObj::injectStageImpl(
        const Stage& parent,
        const Stage& child) {
    //
    // Check the parent and child was not already injected.
    //

    IE_ASSERT(parent->_parentStageEdge == nullptr);

    IE_ASSERT(child->_parentStageEdge == nullptr);
    IE_ASSERT(child->_injectedStageEdge == nullptr);

    //
    // New Edge affects the Stage order.
    //

    _resetStageOrder = true;

    //
    // Create new Edge.
    //

    std::shared_ptr<InjectionEdge> edge(new InjectionEdge);

    edge->_parent = parent;
    edge->_child = child->shared_from_this();
    edge->_model = this;

    edge->_ptrPosInModel = _stageEdgePtrList.emplace(_stageEdgePtrList.end(), edge);

    parent->_injectedStageEdge = edge;
    child->_parentStageEdge = edge;

    //
    // Redirect child inputs to parent.
    //

    for (const auto& childInEdge : child->_inputEdges) {
        if (childInEdge->_input->_producerEdge != nullptr) {
            auto it1 = childInEdge->_input->_producerEdge->_producer->_nextStages.find(childInEdge->_consumer);
            IE_ASSERT(it1 != childInEdge->_input->_producerEdge->_producer->_nextStages.end());
            --it1->second;
            if (it1->second <= 0) {
                childInEdge->_input->_producerEdge->_producer->_nextStages.erase(it1);
            }

            auto it2 = childInEdge->_consumer->_prevStages.find(childInEdge->_input->_producerEdge->_producer);
            IE_ASSERT(it2 != childInEdge->_consumer->_prevStages.end());
            --it2->second;
            if (it2->second <= 0) {
                childInEdge->_consumer->_prevStages.erase(it2);
            }
        }

        childInEdge->_input->_consumerEdges.erase(childInEdge);

        auto parentInEdge = addStageInput(parent, childInEdge->_input);

        childInEdge->_parentEdge = parentInEdge;
        parentInEdge->_childEdge = childInEdge;
    }

    //
    // Redirect child outputs to parent.
    //

    for (const auto& childOutEdge : child->_outputEdges) {
        for (const auto& consumerEdge : childOutEdge->_output->_consumerEdges) {
            auto it1 = consumerEdge->_consumer->_prevStages.find(childOutEdge->_producer);
            IE_ASSERT(it1 != consumerEdge->_consumer->_prevStages.end());
            --it1->second;
            if (it1->second <= 0) {
                consumerEdge->_consumer->_prevStages.erase(it1);
            }

            auto it2 = childOutEdge->_producer->_nextStages.find(consumerEdge->_consumer);
            IE_ASSERT(it2 != childOutEdge->_producer->_nextStages.end());
            --it2->second;
            if (it2->second <= 0) {
                childOutEdge->_producer->_nextStages.erase(it2);
            }
        }

        childOutEdge->_output->_producerEdge = nullptr;

        auto parentOutEdge = addStageOutput(parent, childOutEdge->_output);

        childOutEdge->_parentEdge = parentOutEdge;
        parentOutEdge->_childEdge = childOutEdge;
    }

    //
    // Redirect child temp buffers to parent.
    //

    for (const auto& childEdge : child->_tempBufferEdges) {
        childEdge->_tempBuffer->_tempBufferEdge = nullptr;

        std::shared_ptr<StageTempBufferEdge> parentEdge(new StageTempBufferEdge);

        parentEdge->_stage = parent;
        parentEdge->_tempBuffer = childEdge->_tempBuffer;
        parentEdge->_portInd = parent->_tempBufferEdges.size();
        parentEdge->_model = this;

        parentEdge->_ptrPosInModel = _tempBufferEdgePtrList.emplace(_tempBufferEdgePtrList.end(), parentEdge);

        parent->_tempBufferEdges.emplace_back(parentEdge);
        childEdge->_tempBuffer->_tempBufferEdge = parentEdge;

        childEdge->_parentEdge = parentEdge;
        parentEdge->_childEdge = childEdge;
    }

    //
    // Move child Stage from the Model to parent Stage.
    //

    IE_ASSERT(child->_ptrPosInModel != _stagePtrList.end());
    _stagePtrList.erase(child->_ptrPosInModel);
    child->_ptrPosInModel = _stagePtrList.end();

    _initialStages.erase(child);

    if (parent->_prevStages.empty()) {
        _initialStages.emplace(parent);
    } else {
        _initialStages.erase(parent);
    }

    return edge;
}

void ModelObj::revertInjection(const Injection& edge) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(edge->_model.get() == this);

    auto parentStage = edge->_parent;
    auto childStage = edge->_child;

    IE_ASSERT(parentStage->_model.get() == this);
    IE_ASSERT(childStage->_model.get() == this);
    IE_ASSERT(childStage->_parentStageEdge == edge);

    //
    // The revert affects the Stage order.
    //

    _resetStageOrder = true;

    //
    // Move child Stage from parent Stage to the Model.
    //

    childStage->_ptrPosInModel = _stagePtrList.emplace(_stagePtrList.end(), childStage);

    //
    // Remove Injection Edge from parent and child Stage.
    //

    parentStage->_injectedStageEdge = nullptr;
    childStage->_parentStageEdge = nullptr;

    //
    // Remove Injected Input Edges from parent Stage.
    //

    int startInd = -1;
    int endInd = -1;

    for (const auto& inEdge : parentStage->_inputEdges) {
        if (inEdge->_childEdge == nullptr) {
            IE_ASSERT(startInd < 0);
            continue;
        }

        if (startInd >= 0 && endInd >= 0) {
            IE_ASSERT(inEdge->_childEdge->_consumer != childStage);
        }

        if (inEdge->_childEdge->_consumer != childStage) {
            if (startInd >= 0 && endInd < 0) {
                endInd = inEdge->_portInd;
            }
            continue;
        }

        if (startInd < 0) {
            startInd = inEdge->_portInd;
        }
        if (inEdge->_portInd == parentStage->_inputEdges.size() - 1) {
            endInd = inEdge->_portInd + 1;
        }

        if (inEdge->_input->_producerEdge != nullptr) {
            auto it1 = inEdge->_input->_producerEdge->_producer->_nextStages.find(inEdge->_consumer);
            IE_ASSERT(it1 != inEdge->_input->_producerEdge->_producer->_nextStages.end());
            --it1->second;
            if (it1->second <= 0) {
                inEdge->_input->_producerEdge->_producer->_nextStages.erase(it1);
            }

            auto it2 = inEdge->_consumer->_prevStages.find(inEdge->_input->_producerEdge->_producer);
            IE_ASSERT(it2 != inEdge->_consumer->_prevStages.end());
            --it2->second;
            if (it2->second <= 0) {
                inEdge->_consumer->_prevStages.erase(it2);
            }
        }

        if (inEdge->_childEdge->_input->_producerEdge != nullptr) {
            IE_ASSERT(inEdge->_childEdge->_consumer->_parentStageEdge == nullptr);
            IE_ASSERT(inEdge->_childEdge->_input->_producerEdge->_producer->_parentStageEdge == nullptr);
            ++inEdge->_childEdge->_input->_producerEdge->_producer->_nextStages[inEdge->_childEdge->_consumer];
            ++inEdge->_childEdge->_consumer->_prevStages[inEdge->_childEdge->_input->_producerEdge->_producer];
        }

        inEdge->_childEdge->_parentEdge = nullptr;
        inEdge->_input->_consumerEdges.erase(inEdge);
        inEdge->_input->_consumerEdges.push_back(inEdge->_childEdge);

        IE_ASSERT(inEdge->_ptrPosInModel != _inEdgePtrList.end());
        _inEdgePtrList.erase(inEdge->_ptrPosInModel);
    }

    IE_ASSERT(startInd >= 0 && endInd > startInd && startInd <= parentStage->_inputEdges.size());
    parentStage->_inputEdges.erase(
        parentStage->_inputEdges.begin() + startInd,
        parentStage->_inputEdges.begin() + endInd);

    for (int i = 0; i < parentStage->_inputEdges.size(); ++i) {
        parentStage->_inputEdges[i]->_portInd = i;
    }

    //
    // Remove Injected Output Edges from parent Stage.
    //

    startInd = -1;
    endInd = -1;

    for (const auto& outEdge : parentStage->_outputEdges) {
        if (outEdge->_childEdge == nullptr) {
            IE_ASSERT(startInd < 0);
            continue;
        }

        if (startInd >= 0 && endInd >= 0) {
            IE_ASSERT(outEdge->_childEdge->_producer != childStage);
        }

        if (outEdge->_childEdge->_producer != childStage) {
            if (startInd >= 0 && endInd < 0) {
                endInd = outEdge->_portInd;
            }
            continue;
        }

        if (startInd < 0) {
            startInd = outEdge->_portInd;
        }
        if (outEdge->_portInd == parentStage->_outputEdges.size() - 1) {
            endInd = outEdge->_portInd + 1;
        }

        for (const auto& consumerEdge : outEdge->_output->_consumerEdges) {
            auto it1 = consumerEdge->_consumer->_prevStages.find(outEdge->_producer);
            IE_ASSERT(it1 != consumerEdge->_consumer->_prevStages.end());
            --it1->second;
            if (it1->second <= 0) {
                consumerEdge->_consumer->_prevStages.erase(it1);
            }

            auto it2 = outEdge->_producer->_nextStages.find(consumerEdge->_consumer);
            IE_ASSERT(it2 != outEdge->_producer->_nextStages.end());
            --it2->second;
            if (it2->second <= 0) {
                outEdge->_producer->_nextStages.erase(it2);
            }
        }

        for (const auto& consumerEdge : outEdge->_childEdge->_output->_consumerEdges) {
            IE_ASSERT(outEdge->_childEdge->_producer->_parentStageEdge == nullptr);
            IE_ASSERT(consumerEdge->_consumer->_parentStageEdge == nullptr);
            ++consumerEdge->_consumer->_prevStages[outEdge->_childEdge->_producer];
            ++outEdge->_childEdge->_producer->_nextStages[consumerEdge->_consumer];
        }

        outEdge->_childEdge->_parentEdge = nullptr;
        outEdge->_output->_producerEdge = outEdge->_childEdge;

        IE_ASSERT(outEdge->_ptrPosInModel != _outEdgePtrList.end());
        _outEdgePtrList.erase(outEdge->_ptrPosInModel);
    }

    IE_ASSERT(startInd >= 0 && endInd > startInd && startInd <= parentStage->_outputEdges.size());
    parentStage->_outputEdges.erase(
        parentStage->_outputEdges.begin() + startInd,
        parentStage->_outputEdges.begin() + endInd);

    for (int i = 0; i < parentStage->_outputEdges.size(); ++i) {
        parentStage->_outputEdges[i]->_portInd = i;
    }

    //
    // Remove Injected Temp Buffer Edges from parent Stage.
    //

    startInd = -1;
    endInd = -1;

    for (const auto& tempBufferEdge : parentStage->_tempBufferEdges) {
        if (tempBufferEdge->_childEdge == nullptr) {
            IE_ASSERT(startInd < 0);
            continue;
        }

        if (startInd >= 0 && endInd >= 0) {
            IE_ASSERT(tempBufferEdge->_childEdge->_stage != childStage);
        }

        if (tempBufferEdge->_childEdge->_stage != childStage) {
            if (startInd >= 0 && endInd < 0) {
                endInd = tempBufferEdge->_portInd;
            }
            continue;
        }

        if (startInd < 0) {
            startInd = tempBufferEdge->_portInd;
        }
        if (tempBufferEdge->_portInd == parentStage->_tempBufferEdges.size() - 1) {
            endInd = tempBufferEdge->_portInd + 1;
        }

        tempBufferEdge->_childEdge->_parentEdge = nullptr;
        tempBufferEdge->_tempBuffer->_tempBufferEdge = tempBufferEdge->_childEdge;

        IE_ASSERT(tempBufferEdge->_ptrPosInModel != _tempBufferEdgePtrList.end());
        _tempBufferEdgePtrList.erase(tempBufferEdge->_ptrPosInModel);
    }

    if (startInd >= 0) {
        IE_ASSERT(endInd > startInd && startInd <= parentStage->_tempBufferEdges.size());
        parentStage->_tempBufferEdges.erase(
            parentStage->_tempBufferEdges.begin() + startInd,
            parentStage->_tempBufferEdges.begin() + endInd);

        for (int i = 0; i < parentStage->_tempBufferEdges.size(); ++i) {
            parentStage->_tempBufferEdges[i]->_portInd = i;
        }
    }

    if (parentStage->_prevStages.empty()) {
        _initialStages.emplace(parentStage);
    } else {
        _initialStages.erase(parentStage);
    }

    if (childStage->_prevStages.empty()) {
        _initialStages.emplace(childStage);
    } else {
        _initialStages.erase(childStage);
    }

    //
    // Remove the Injection Edge from the Model.
    //

    IE_ASSERT(edge->_ptrPosInModel != _stageEdgePtrList.end());
    _stageEdgePtrList.erase(edge->_ptrPosInModel);
}

ModelObj::DataToDataEdgeHelper::~DataToDataEdgeHelper() {
    //
    // Check that `done` was called.
    //

    if (_model != nullptr) {
        std::terminate();
    }
}

ModelObj::DataToDataEdgeHelper& ModelObj::DataToDataEdgeHelper::parent(const Data& parent) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `parent` was not called.
    //

    IE_ASSERT(_parent == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(parent->_model == _model);

    _parent = parent;

    return *this;
}

ModelObj::DataToDataEdgeHelper& ModelObj::DataToDataEdgeHelper::child(const Data& child) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `child` was not called.
    //

    IE_ASSERT(_child == nullptr);

    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(child->_model == _model);

    _child = child;

    return *this;
}

ModelObj::DataToDataEdgeHelper& ModelObj::DataToDataEdgeHelper::mode(SharedDataMode mode) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `mode` was not called.
    //

    IE_ASSERT(!_modeSet);

    _mode = mode;
    _modeSet = true;

    return *this;
}

ModelObj::DataToDataEdgeHelper& ModelObj::DataToDataEdgeHelper::order(SharedDataOrder order) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `order` was not called.
    //

    IE_ASSERT(!_orderSet);

    _order = order;
    _orderSet = true;

    return *this;
}

ModelObj::DataToDataEdgeHelper& ModelObj::DataToDataEdgeHelper::offset(const DimValues& offset) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `offset` was not called.
    //

    IE_ASSERT(!_offsetSet);

    _offset = offset;
    _offsetSet = true;

    return *this;
}

ModelObj::DataToDataEdgeHelper& ModelObj::DataToDataEdgeHelper::connectionMode(SharedConnectionMode connectionMode) {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that `offset` was not called.
    //

    IE_ASSERT(!_offsetSet);

    _connectionMode = connectionMode;

    return *this;
}

DataToDataAllocation ModelObj::DataToDataEdgeHelper::done() {
    //
    // Check that `done` was not called.
    //

    IE_ASSERT(_model != nullptr);

    //
    // Check that all fields were set.
    //

    IE_ASSERT(_parent != nullptr);
    IE_ASSERT(_child != nullptr);
    IE_ASSERT(_modeSet);
    IE_ASSERT(_orderSet);

    AutoScope autoNullModel([&] {
        _model = nullptr;
    });
    //
    // Call the actual implementation.
    //

    auto edge = _model->connectDataWithDataImpl(
        _parent, _child,
        _mode, _order,
        _offset, _connectionMode);

    //
    // Reset internal state.
    //

    _model = nullptr;

    return edge;
}

namespace {

Stage getDataConnectionStage(
        const Data& parent,
        const Data& child,
        SharedDataMode mode,
        SharedDataOrder order,
        const DimValues& offset,
        const Model& model) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(parent->model() == model);
    IE_ASSERT(child->model() == model);

    //
    // Get producer and consumer data.
    //

    Data producer, consumer;
    if (order == SharedDataOrder::ChildWritesToParent) {
        producer = child;
        consumer = parent;
    } else if (order == SharedDataOrder::ParentWritesToChild) {
        producer = parent;
        consumer = child;
    } else {
        VPU_THROW_EXCEPTION << "Invalid data order " << order;
    }

    //
    // Child must be Intermediate.
    //

    VPU_THROW_UNLESS(
        child->usage() == DataUsage::Intermediate,
        "Tried to share memory for non-Intermediate Data node %v with usage %v",
        child, child->usage());

    //
    // Parent can't be Temp or Fake.
    //

    VPU_THROW_UNLESS(
        parent->usage() != DataUsage::Temp && parent->usage() != DataUsage::Fake,
        "Can't share memory for Data node %v with usage %v",
        parent, parent->usage());

    //
    // Consumer must be accesible from the producer.
    //

    Stage connectionStage;

    for (const auto& consumerEdge : producer->consumerEdges()) {
        for (const auto& outEdge : consumerEdge->consumer()->outputEdges()) {
            if (outEdge->output() == consumer) {
                connectionStage = consumerEdge->consumer();
                break;
            }
        }

        if (connectionStage != nullptr) {
            break;
        }
    }

    IE_ASSERT(connectionStage != nullptr);

    IE_ASSERT(connectionStage->model() == model);

    //
    // Connection stage must be special.
    //

    VPU_THROW_UNLESS(
        connectionStage->category() == StageCategory::Special,
        "Invalid category %v for connection Stage node %v between Data node %v (parent) and Data node %v (child) for sharing memory",
        connectionStage->category(), connectionStage, parent, child);

    //
    // Special checks for each mode.
    //

    if (mode == SharedDataMode::ROI) {
        //
        // Check connection stage type and that parent has the largest buffer.
        //

        if (connectionStage->type() == StageType::StubConcat ||
            connectionStage->type() == StageType::Expand) {
            IE_ASSERT(producer == child);
            IE_ASSERT(consumer == parent);
        } else if (connectionStage->type() == StageType::Split ||
                   connectionStage->type() == StageType::Crop) {
            IE_ASSERT(producer == parent);
            IE_ASSERT(consumer == child);
        } else {
            VPU_THROW_EXCEPTION
                    << "Stage type " << connectionStage->type()
                    << " can't be used for ROI data connection";
        }

        //
        // Parent and child must have the same order.
        //

        VPU_THROW_UNLESS(
            parent->desc().dimsOrder() == child->desc().dimsOrder(),
            "Parent Data node %v and child Data node %v have different DimsOrder (%v vs %v), not appicable for ROI mode",
            parent, child, parent->desc().dimsOrder(), child->desc().dimsOrder());

        //
        // Offset must be valid.
        //

        for (const auto& p : offset) {
            IE_ASSERT(parent->desc().dimsOrder().hasDim(p.first));

            IE_ASSERT(child->desc().dim(p.first) + p.second <= parent->desc().dim(p.first));
        }

        //
        // Check strides requirements
        //

        VPU_INTERNAL_CHECK(
            checkStrides(child->desc(), parent->strides(), child->requiredStrides()),
            "Strides requirements mismatch between parent Data node %v and child Data node %v",
            parent, child);
    } else if (mode == SharedDataMode::Reshape) {
        //
        // Check connection stage type.
        //

        IE_ASSERT(connectionStage->type() == StageType::Reshape);

        //
        // Parent and child must have the same data type.
        //

        IE_ASSERT(parent->desc().type() == child->desc().type());

        //
        // Parent and child must have the same number of elements.
        //

        IE_ASSERT(parent->desc().totalDimSize() == child->desc().totalDimSize());

        //
        // Parent and child must be compact.
        //

        // TODO: can we weaken this restriction?
        IE_ASSERT(parent->checkStrides(StridesRequirement::compact()));
        IE_ASSERT(child->checkStrides(StridesRequirement::compact()));
    } else {
        VPU_THROW_EXCEPTION << "Invalid shared data mode " << mode;
    }

    return connectionStage;
}

}  // namespace

DataToDataAllocation ModelObj::connectDataWithDataImpl(
        const Data& parent,
        const Data& child,
        SharedDataMode mode,
        SharedDataOrder order,
        const DimValues& offset,
        SharedConnectionMode connectionMode) {
    //
    // Child must not have other parents
    //

    IE_ASSERT(child->parentDataToDataEdge() == nullptr);

    //
    // Create new Edge.
    //

    std::shared_ptr<DataToDataAllocationEdge> edge(new DataToDataAllocationEdge);
    edge->_ptrPosInModel = _dataEdgePtrList.emplace(_dataEdgePtrList.end(), edge);

    edge->_parent = parent;
    edge->_child = child;
    edge->_connectionMode = connectionMode;
    if (connectionMode == SharedConnectionMode::SINGLE_STAGE) {
        edge->_connection = getDataConnectionStage(
            parent, child,
            mode, order, offset,
            this);
    }
    edge->_mode = mode;
    edge->_order = order;

    if (mode == SharedDataMode::ROI) {
        edge->attrs().set("offset", offset);
    }

    parent->_childDataToDataEdges.push_back(edge);
    child->_parentDataToDataEdge = edge;

    //
    // Notify allocator.
    //

    if (parent->usage() != DataUsage::Intermediate) {
        getAllocator().setNeedToAllocNonIntermData();
    }

    return edge;
}

namespace {

bool isStageDependencyNeeded(
        const Stage& dependentStage,
        const Data& dependency) {
    const auto& dependencyProducer = dependency->producer();

    if (dependencyProducer == nullptr) {
        return false;
    }

    if (dependentStage == dependencyProducer) {
        return false;
    }

    // Check one level above, it covers current cases while checking all the levels might be computationally expensive
    for (const auto& prevStage : dependencyProducer->prevStages()) {
        if (prevStage == dependentStage) {
            return false;
        }
    }

    for (const auto& prevStage : dependentStage->prevStages()) {
        if (prevStage == dependencyProducer) {
            return false;
        }
    }

    return true;
}

} // namespace

DataToShapeAllocation ModelObj::connectDataWithShape(
        const Data& parent,
        const Data& child) {
    VPU_THROW_UNLESS(child->parentDataToShapeEdge() == nullptr,
        "connectDataWithShape failed: child data {} with usage {} must not have any parents "
        "but it actually have (data {} with usage {})",
        child->name(), child->usage(), child->parentDataToShapeEdge()->parent()->name(), child->parentDataToShapeEdge()->parent()->usage());

    std::shared_ptr<DataToShapeAllocationEdge> edge(new DataToShapeAllocationEdge);
    edge->_ptrPosInModel = _shapeEdgePtrList.emplace(_shapeEdgePtrList.end(), edge);

    edge->_parent = parent;
    edge->_child = child;

    parent->_childDataToShapeEdges.push_back(edge);
    child->_parentDataToShapeEdge = edge;

    const auto& childProducer = child->producer();

    if (childProducer && isStageDependencyNeeded(childProducer, parent)) {
        // Shape and data are produced from different stages, make sure that shape is calculated before data
        addStageDependency(childProducer, parent);
    }

    return edge;
}

void ModelObj::replaceDataToShapeParent(
        const DataToShapeAllocation& edge,
        const Data& newParent) {
    const auto oldParent = edge->parent();
    const auto child = edge->child();

    oldParent->_childDataToShapeEdges.erase(edge);
    edge->_parent = newParent;
    newParent->_childDataToShapeEdges.push_back(edge);

    const auto& childProducer = child->producer();
    if (childProducer != nullptr) {
        removeStageDependency(childProducer, oldParent);

        if (isStageDependencyNeeded(childProducer, newParent)) {
            // Shape and data are produced from different stages, make sure that shape is calculated before data
            addStageDependency(childProducer, newParent);
        }
    }
}

void ModelObj::replaceDataToShapeChild(
        const DataToShapeAllocation& edge,
        const Data& newChild) {
    const auto parent = edge->parent();
    const auto oldChild = edge->child();

    oldChild->_parentDataToShapeEdge = nullptr;
    edge->_child = newChild;

    VPU_THROW_UNLESS(newChild->_parentDataToShapeEdge == nullptr,
        "replaceDataToShapeChild failed: newChild {} with usage {} already has parent {} with usage {}",
        newChild->name(), newChild->usage(), newChild->_parentDataToShapeEdge->parent()->name(), newChild->_parentDataToShapeEdge->parent()->usage());

    newChild->_parentDataToShapeEdge = edge;

    const auto& oldChildProducer = oldChild->producer();
    if (oldChildProducer != nullptr) {
        removeStageDependency(oldChildProducer, parent);
    }

    const auto& newChildProducer = newChild->producer();

    if (newChildProducer && isStageDependencyNeeded(newChildProducer, parent)) {
        // Shape and data are produced from different stages, make sure that shape is calculated before data
        addStageDependency(newChildProducer, parent);
    }
}

void ModelObj::replaceDataToDataParent(
        const DataToDataAllocation& edge,
        const Data& newParent) {
    auto oldParent = edge->parent();
    auto child = edge->child();

    oldParent->_childDataToDataEdges.erase(edge);

    edge->_parent = newParent;
    if (edge->connectionMode() == SharedConnectionMode::SINGLE_STAGE) {
        edge->_connection = getDataConnectionStage(
            newParent, child,
            edge->mode(), edge->order(),
            edge->attrs().getOrDefault<DimValues>("offset"),
            this);
    }

    newParent->_childDataToDataEdges.push_back(edge);

    if (oldParent->usage() != DataUsage::Intermediate ||
        newParent->usage() != DataUsage::Intermediate) {
        getAllocator().setNeedToAllocNonIntermData();
    }
}

void ModelObj::replaceDataToDataChild(
        const DataToDataAllocation& edge,
        const Data& newChild) {
    auto parent = edge->parent();
    auto oldChild = edge->child();

    oldChild->_parentDataToDataEdge = nullptr;

    edge->_child = newChild;
    if (edge->connectionMode() == SharedConnectionMode::SINGLE_STAGE) {
        edge->_connection = getDataConnectionStage(
            parent, newChild,
            edge->mode(), edge->order(),
            edge->attrs().getOrDefault<DimValues>("offset"),
            this);
    }

    newChild->_parentDataToDataEdge = edge;

    if (parent->usage() != DataUsage::Intermediate) {
        getAllocator().setNeedToAllocNonIntermData();
    }
}

void ModelObj::disconnectDatas(const DataToDataAllocation& edge) {
    auto parent = edge->parent();
    auto child = edge->child();

    child->_parentDataToDataEdge = nullptr;
    parent->_childDataToDataEdges.erase(edge);

    IE_ASSERT(edge->_ptrPosInModel != _dataEdgePtrList.end());
    _dataEdgePtrList.erase(edge->_ptrPosInModel);

    if (parent->usage() != DataUsage::Intermediate) {
        getAllocator().setNeedToAllocNonIntermData();
    }
}

void ModelObj::disconnectDatas(const DataToShapeAllocation& edge) {
    auto parent = edge->parent();
    auto child = edge->child();

    child->_parentDataToShapeEdge = nullptr;
    parent->_childDataToShapeEdges.erase(edge);

    VPU_THROW_UNLESS(edge->_ptrPosInModel != _shapeEdgePtrList.end(),
        "disconnect Datas (parent {} with usage {} and child {} with usage {}) with DataToShape connection failed: "
        "no such edge in Model's DataToShapeEdges list", parent->name(), parent->usage(), child->name(), child->usage());

    _shapeEdgePtrList.erase(edge->_ptrPosInModel);

    const auto& childProducer = child->producer();
    if (childProducer != nullptr) {
        removeStageDependency(childProducer, parent);
    }
}

void ModelObj::disconnectStage(const Stage& stage) {
    //
    // Check that objects belong to the same Model.
    //

    IE_ASSERT(stage->_model.get() == this);

    //
    // This affect the Stage order.
    //

    _resetStageOrder = true;

    //
    // Disconnect input datas.
    //

    for (const auto& inEdge : stage->_inputEdges) {
        if (inEdge->_input->_producerEdge != nullptr) {
            auto it1 = inEdge->_input->_producerEdge->_producer->_nextStages.find(inEdge->_consumer);
            IE_ASSERT(it1 != inEdge->_input->_producerEdge->_producer->_nextStages.end());
            --it1->second;
            if (it1->second <= 0) {
                inEdge->_input->_producerEdge->_producer->_nextStages.erase(it1);
            }

            auto it2 = inEdge->_consumer->_prevStages.find(inEdge->_input->_producerEdge->_producer);
            IE_ASSERT(it2 != inEdge->_consumer->_prevStages.end());
            --it2->second;
            if (it2->second <= 0) {
                inEdge->_consumer->_prevStages.erase(it2);
            }
        }

        inEdge->_input->_consumerEdges.erase(inEdge);

        IE_ASSERT(inEdge->_ptrPosInModel != _inEdgePtrList.end());
        _inEdgePtrList.erase(inEdge->_ptrPosInModel);
    }

    stage->_inputEdges.clear();

    //
    // Disconnect output datas.
    //

    for (const auto& outEdge : stage->_outputEdges) {
        // Disconnect from dependency
        if (const auto& dataToShapeEdge = outEdge->output()->parentDataToShapeEdge()) {
            removeStageDependency(stage, dataToShapeEdge->parent());
        }
        // Disconnect from consumers
        for (const auto& consumerEdge : outEdge->_output->_consumerEdges) {
            auto it1 = consumerEdge->_consumer->_prevStages.find(outEdge->_producer);
            IE_ASSERT(it1 != consumerEdge->_consumer->_prevStages.end());
            --it1->second;
            if (it1->second <= 0) {
                consumerEdge->_consumer->_prevStages.erase(it1);
            }

            auto it2 = outEdge->_producer->_nextStages.find(consumerEdge->_consumer);
            IE_ASSERT(it2 != outEdge->_producer->_nextStages.end());
            --it2->second;
            if (it2->second <= 0) {
                outEdge->_producer->_nextStages.erase(it2);
            }
        }

        outEdge->_output->_producerEdge = nullptr;

        IE_ASSERT(outEdge->_ptrPosInModel != _outEdgePtrList.end());
        _outEdgePtrList.erase(outEdge->_ptrPosInModel);
    }

    stage->_outputEdges.clear();

    //
    // Disconnect temp datas.
    //

    for (const auto& tempBufferEdge : stage->_tempBufferEdges) {
        tempBufferEdge->_tempBuffer->_tempBufferEdge = nullptr;

        IE_ASSERT(tempBufferEdge->_ptrPosInModel != _tempBufferEdgePtrList.end());
        _tempBufferEdgePtrList.erase(tempBufferEdge->_ptrPosInModel);
    }

    stage->_tempBufferEdges.clear();

    _allocator.setNeedToAllocNonIntermData();
}

void ModelObj::removeStage(const Stage& stage) {
    IE_ASSERT(stage->_model.get() == this);

    _resetStageOrder = true;

    disconnectStage(stage);

    _initialStages.erase(stage);

    IE_ASSERT(stage->_ptrPosInModel != _stagePtrList.end());
    _stagePtrList.erase(stage->_ptrPosInModel);
}

void ModelObj::cleanUp() {
    for (const auto& data : datas()) {
        if (data->_usage == DataUsage::Input) {
            VPU_THROW_UNLESS(!data->_consumerEdges.empty() || !data->childDataToShapeEdges().empty(),
                    "Input data {} must either have at least one consumer (but got zero) or be a shape data.", data->name());
            IE_ASSERT(data->_parentDataToDataEdge == nullptr);
        } else if (data->_usage == DataUsage::Output) {
            IE_ASSERT(data->_producerEdge != nullptr);
            IE_ASSERT(data->_parentDataToDataEdge == nullptr);
        } else if (data->_usage == DataUsage::Temp) {
            if (data->_tempBufferEdge == nullptr) {
                _dataList.erase(data);

                IE_ASSERT(data->_ptrPosInModel != _dataPtrList.end());
                _dataPtrList.erase(data->_ptrPosInModel);
            }
        } else {
            if (data->_consumerEdges.empty() && data->_producerEdge == nullptr) {
                removeUnusedData(data);
            }
        }
    }
}

void ModelObj::buildStageOrder() const {
    if (!_resetStageOrder) {
        IE_ASSERT(_orderedStageList.size() == _stagePtrList.size());
        return;
    }

    VPU_PROFILE(buildStageOrder);

    _orderedStageList.clear();
    _resetStageOrder = false;

    if (_stagePtrList.empty()) {
        return;
    }

    //
    // Run recursive DFS algorithm
    //

    IE_ASSERT(!_initialStages.empty());

    StageMap<bool> visitedMap;

    // Traverse input Stages in reverse order, because the algorithm uses push_front.
    // With reverse order at the loop we will get original order in result.
    for (const auto& stage : _initialStages | asRange() | reverse()) {
        runDFS(stage, visitedMap);
    }

    IE_ASSERT(_orderedStageList.size() == _stagePtrList.size());

    int stageInd = 0;
    for (const auto& stage : _orderedStageList) {
        stage->_index = stageInd;
        ++stageInd;
    }
}

void ModelObj::reorderStages(
        const StageComparator& comparator) {
    _nextStagesComparator = comparator;
    _resetStageOrder = true;
}

void ModelObj::setStagesOrder(const Stage& parent, const Stage& child) {
    ++parent->_nextStages[child];
    ++child->_prevStages[parent];
    _initialStages.erase(child);
}

void ModelObj::removeStagesOrder(const Stage& parent, const Stage& child) {
    auto parentNextStage = parent->_nextStages.find(child);
    VPU_THROW_UNLESS(parentNextStage != parent->_nextStages.end(),
                     "removeStagesOrder failed: parent {} with type {} doesn't have {} with type {} as its next stage",
                     parent->name(), parent->type(), child->name(), child->type());
    --parentNextStage->second;
    if (parentNextStage->second <= 0) {
        parent->_nextStages.erase(parentNextStage);
    }

    auto childPrevStage = child->_prevStages.find(parent);
    VPU_THROW_UNLESS(childPrevStage != child->_prevStages.end(),
                     "removeStagesOrder failed: child {} with type {} doesn't have {} with type {} as its previous stage",
                     child->name(), child->type(), parent->name(), parent->type());
    --childPrevStage->second;
    if (childPrevStage->second <= 0) {
        child->_prevStages.erase(childPrevStage);
    }
    if (child->_prevStages.empty()) {
        _initialStages.emplace(child);
    }
}

void ModelObj::runDFS(
        const Stage& stage,
        StageMap<bool>& visitedMap) const {
    IE_ASSERT(stage->_parentStageEdge == nullptr);

    visitedMap[stage] = false;

    auto nextStages = stage->nextStages() | asSmallVector();
    if (_nextStagesComparator)
        std::sort(nextStages.begin(), nextStages.end(), _nextStagesComparator);

    // Traverse next Stages in reverse order, because the algorithm uses push_front.
    // With reverse order at the loop we will get original order in result.
    for (const auto& nextStage : nextStages | asRange() | reverse()) {
        auto it = visitedMap.find(nextStage);

        if (it != visitedMap.end()) {
            auto visited = it->second;

            if (!visited) {
                VPU_THROW_EXCEPTION << "Graph has cycle";
            }

            continue;
        }

        runDFS(nextStage, visitedMap);
    }

    visitedMap[stage] = true;

    _orderedStageList.push_front(stage);
}

Stage ModelObj::addNewStageImpl(
    const std::string& name,
    StageType type,
    const ie::CNNLayerPtr& origLayer,
    const DataVector& inputs,
    const DataVector& outputs,
    const FuncRef<StagePtr()>& creator) {
    //
    // Check that Stage has inputs and outputs.
    //

    IE_ASSERT(!inputs.empty());
    IE_ASSERT(!outputs.empty() || type == StageType::None);

    //
    // Check that Data objects belong to the same Model.
    //

    for (const auto& input : inputs) {
        IE_ASSERT(input->_model.get() == this);
    }
    for (const auto& output : outputs) {
        IE_ASSERT(output->_model.get() == this);
    }

    //
    // Check that there are no loops.
    //

    // TODO: more advanced check.
    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            IE_ASSERT(input != output);
        }
    }

    _resetStageOrder = true;

    auto stage = creator();

    stage->_name      = name;
    stage->_id        = _stagesIdCount++;
    stage->_type      = type;
    stage->_origLayer = origLayer;
    stage->_model     = this;

    for (const auto& input : inputs) {
        addStageInput(stage, input);
    }
    for (const auto& output : outputs) {
        addStageOutput(stage, output);
    }

    stage->_ptrPosInModel = _stagePtrList.emplace(_stagePtrList.end(), stage);

    return stage;
}

void ModelObj::removeUnusedData(const Data& data) {
    VPU_INTERNAL_CHECK(
       data->numConsumers() == 0,
       "Data node %v was mistakenly classified as unused, while it has %v consumers",
        data, data->numConsumers());

    VPU_INTERNAL_CHECK(
       data->_ptrPosInModel != _dataPtrList.end(),
       "Tried to remove Data node %v, which doesn't belong to current Model %v",
       data, name());

    if (data->usage() != DataUsage::Intermediate &&
        data->usage() != DataUsage::Temp) {
        _allocator.setNeedToAllocNonIntermData();
    }

    if (const auto dataToShapeEdge = data->parentDataToShapeEdge()) {
        const auto shape = dataToShapeEdge->parent();
        disconnectDatas(dataToShapeEdge);
        VPU_INTERNAL_CHECK(!shape->childDataToShapeEdges().empty() || !shape->consumerEdges().empty(),
                "Removed unused data (with name {}) must have a shape data (with name {}) which is a shape "
                "for other data or has consumer", data->name(), shape->name());
    }

    _dataList.erase(data);
    _dataPtrList.erase(data->_ptrPosInModel);
}

}  // namespace vpu
