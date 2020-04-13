// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/special_stage_processor.hpp"

#include <vector>
#include <set>
#include <utility>

namespace vpu {

void SpecialStageProcessor::processSplit(
        const Model& model,
        const Stage& stage) {
    IE_ASSERT(stage->type() == StageType::Split);

    auto input = stage->input(0);

    const auto& offsets = stage->attrs().get<std::vector<DimValues>>("offsets");
    IE_ASSERT(offsets.size() == checked_cast<size_t>(stage->numOutputs()));

    for (const auto& outEdge : stage->outputEdges()) {
        IE_ASSERT(outEdge->portInd() >= 0);
        IE_ASSERT(checked_cast<size_t>(outEdge->portInd()) < offsets.size());

        auto output = outEdge->output();
        const auto& offsetFromInput = offsets[checked_cast<size_t>(outEdge->portInd())];

        IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());
        IE_ASSERT(offsetFromInput.size() <= checked_cast<size_t>(input->desc().numDims()));
        for (const auto& p : offsetFromInput) {
            IE_ASSERT(input->desc().dimsOrder().hasDim(p.first));
            IE_ASSERT(p.second + output->desc().dim(p.first) <= input->desc().dim(p.first));
        }

        //
        // Check if we need to insert Copy stage
        //

        bool needCopy = false;
        if (output->usage() != DataUsage::Intermediate) {
            needCopy = true;
        } else if (output->parentDataEdge() != nullptr) {
            needCopy = true;
        } else {
            //
            // Check output StridesRequirement.
            //

            IE_ASSERT(output->checkStrides(output->requiredStrides()));
            if (!checkStrides(output->desc(), input->strides(), output->requiredStrides())) {
                needCopy = true;
            }

            //
            // Check consumers StridesRequirement.
            //

            if (!needCopy) {
                for (const auto& consumerEdge : output->consumerEdges()) {
                    const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

                    if (consumerInfo.hasInput(consumerEdge)) {
                        const auto& consumerStrideReqs = consumerInfo.getInput(consumerEdge);
                        IE_ASSERT(output->checkStrides(consumerStrideReqs));

                        if (!checkStrides(output->desc(), input->strides(), consumerStrideReqs)) {
                            needCopy = true;
                            break;
                        }
                    }
                }
            }
        }

        //
        // Insert Copy if needed
        //

        if (needCopy) {
            auto outputCopy = model->duplicateData(output, "@copy");
            outputCopy->resetRequiredStrides();

            auto outPortInd = outEdge->portInd();

            model->replaceStageOutput(outEdge, outputCopy);

            auto copyStage = _stageBuilder->addCopyStage(
                model,
                formatString("%s@output=%d@copy-for-split", stage->name(), outPortInd),
                stage->origLayer(),
                outputCopy,
                output,
                "special::split");
            if (stage->attrs().has("batchInd")) {
                copyStage->attrs().set("batchInd", stage->attrs().get<int>("batchInd"));
            }

            output = outputCopy;
        }

        //
        // Add Data<->Data edge
        //

        model->connectDatas()
            .parent(input)
            .child(output)
            .mode(SharedDataMode::ROI)
            .order(SharedDataOrder::ParentWritesToChild)
            .offset(offsetFromInput)
            .done();
    }
}

void SpecialStageProcessor::processConcat(
        const Model& model,
        const Stage& stage) {
    auto output = stage->output(0);

    const auto& offsets = stage->attrs().get<std::vector<DimValues>>("offsets");
    IE_ASSERT(offsets.size() == checked_cast<size_t>(stage->numInputs()));

    for (const auto& inEdge : stage->inputEdges()) {
        IE_ASSERT(inEdge->portInd() >= 0);
        IE_ASSERT(checked_cast<size_t>(inEdge->portInd()) < offsets.size());

        auto input = inEdge->input();
        const auto& offsetFromOutput = offsets[checked_cast<size_t>(inEdge->portInd())];

        IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());
        IE_ASSERT(offsetFromOutput.size() <= checked_cast<size_t>(output->desc().numDims()));
        for (const auto& p : offsetFromOutput) {
            IE_ASSERT(output->desc().dimsOrder().hasDim(p.first));
            IE_ASSERT(p.second + input->desc().dim(p.first) <= output->desc().dim(p.first));
        }

        //
        // Check if we need to insert Copy stage
        //

        bool needCopy = false;
        bool optionalCopy = false;
        if (input->usage() != DataUsage::Intermediate) {
            needCopy = true;
            optionalCopy = false;
        } else if (input->parentDataEdge() != nullptr) {
            needCopy = true;
            optionalCopy = false;
        } else {
            //
            // Check input StridesRequirement.
            //

            IE_ASSERT(input->checkStrides(input->requiredStrides()));
            if (!checkStrides(input->desc(), output->strides(), input->requiredStrides())) {
                needCopy = true;
                optionalCopy = false;
            }

            //
            // Check consumers StridesRequirement.
            //

            if (!needCopy) {
                for (const auto& consumerEdge : input->consumerEdges()) {
                    const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

                    if (consumerInfo.hasInput(consumerEdge)) {
                        const auto& consumerStrideReqs = consumerInfo.getInput(consumerEdge);
                        IE_ASSERT(input->checkStrides(consumerStrideReqs));

                        if (!checkStrides(input->desc(), output->strides(), consumerStrideReqs)) {
                            needCopy = true;
                            optionalCopy = false;
                        }
                    }
                }
            }

            //
            // Check producer StridesRequirement.
            //

            if (!needCopy) {
                if (auto producerEdge = input->producerEdge()) {
                    const auto& producerInfo = producerEdge->producer()->getDataStridesRequirements();

                    if (producerInfo.hasOutput(producerEdge)) {
                        const auto& producerStrideReqs = producerInfo.getOutput(producerEdge);
                        IE_ASSERT(input->checkStrides(producerStrideReqs));

                        if (!checkStrides(input->desc(), output->strides(), producerStrideReqs)) {
                            needCopy = true;
                            optionalCopy = false;
                        }
                    }

                    if (!needCopy) {
                        //
                        // To reduce the size of HW output (still can be optimized).
                        //

                        if (producerEdge->producer()->category() == StageCategory::HW) {
                            needCopy = true;
                            optionalCopy = true;
                        }
                    }
                }
            }
        }

        //
        // Insert Copy if needed
        //

        if (needCopy) {
            Data inputCopy;
            if (input->usage() == DataUsage::Const) {
                inputCopy = model->addNewData(
                    input->name() + "@copy",
                    input->desc());
            } else {
                inputCopy = model->duplicateData(
                    input,
                    "@copy");
                inputCopy->resetRequiredStrides();
            }

            auto copyStage = _stageBuilder->addCopyStage(
                model,
                formatString("%s@input=%d@copy-for-concat", stage->name(), inEdge->portInd()),
                stage->origLayer(),
                input,
                inputCopy,
                "special::concat");
            copyStage->attrs().set<bool>("optional", optionalCopy);
            if (stage->attrs().has("batchInd")) {
                copyStage->attrs().set("batchInd", stage->attrs().get<int>("batchInd"));
            }

            model->replaceStageInput(inEdge, inputCopy);

            input = inputCopy;
        }

        //
        // Add Data<->Data edge
        //

        model->connectDatas()
            .parent(output)
            .child(input)
            .mode(SharedDataMode::ROI)
            .order(SharedDataOrder::ChildWritesToParent)
            .offset(offsetFromOutput)
            .done();
    }
}


void SpecialStageProcessor::processReshape(
        const Model& model,
        const Stage& stage) {
    auto input = stage->input(0);
    auto output = stage->output(0);

    IE_ASSERT(input->desc().dimsOrder() == DimsOrder::fromNumDims(input->desc().numDims()));
    IE_ASSERT(input->checkStrides(StridesRequirement::compact()));

    IE_ASSERT(output->desc().dimsOrder() == DimsOrder::fromNumDims(output->desc().numDims()));
    IE_ASSERT(output->checkStrides(StridesRequirement::compact()));

    //
    // Check if we need to insert Copy stage
    //

    bool needCopy = false;
    if (input->usage() != DataUsage::Intermediate &&
        output->usage() != DataUsage::Intermediate) {
        needCopy = true;
    } else if (input->parentDataEdge() != nullptr &&
               output->parentDataEdge() != nullptr) {
        needCopy = true;
    }

    //
    // Insert Copy if needed
    //

    if (needCopy) {
        Data inputCopy;
        if (input->usage() == DataUsage::Const) {
            inputCopy = model->addNewData(
                input->name() + "@copy",
                input->desc());
        } else {
            inputCopy = model->duplicateData(
                input,
                "@copy");
        }
        inputCopy->updateRequiredStrides(StridesRequirement::compact());

        auto copyStage = _stageBuilder->addCopyStage(
            model,
            formatString("%s@copy-for-reshape", stage->name()),
            stage->origLayer(),
            input,
            inputCopy,
            "special::reshape");
        if (stage->attrs().has("batchInd")) {
            copyStage->attrs().set("batchInd", stage->attrs().get<int>("batchInd"));
        }

        model->replaceStageInput(stage->inputEdge(0), inputCopy);

        input = inputCopy;
    }

    //
    // Add Data<->Data edge
    //

    if (input->usage() == DataUsage::Intermediate &&
        input->parentDataEdge() == nullptr) {
        model->connectDatas()
            .parent(output)
            .child(input)
            .mode(SharedDataMode::Reshape)
            .order(SharedDataOrder::ChildWritesToParent)
            .done();
    } else {
        IE_ASSERT(output->usage() == DataUsage::Intermediate);
        IE_ASSERT(output->parentDataEdge() == nullptr);

        model->connectDatas()
            .parent(input)
            .child(output)
            .mode(SharedDataMode::Reshape)
            .order(SharedDataOrder::ParentWritesToChild)
            .done();
    }
}

void SpecialStageProcessor::processExpand(
        const Model& model,
        const Stage& stage) {
    auto input = stage->input(0);
    auto output = stage->output(0);

    const auto& offset = stage->attrs().get<DimValues>("offset");

    IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());

    IE_ASSERT(offset.size() <= checked_cast<size_t>(output->desc().numDims()));
    for (const auto& p : offset) {
        IE_ASSERT(output->desc().dimsOrder().hasDim(p.first));
        IE_ASSERT(p.second + input->desc().dim(p.first) <= output->desc().dim(p.first));
    }

    //
    // Check if we need to insert Copy stage
    //

    bool needCopy = false;
    bool optionalCopy = false;
    if (input->usage() != DataUsage::Intermediate) {
        needCopy = true;
        optionalCopy = false;
    } else if (input->parentDataEdge() != nullptr) {
        needCopy = true;
        optionalCopy = false;
    } else {
        //
        // Check input StridesRequirement.
        //

        IE_ASSERT(input->checkStrides(input->requiredStrides()));
        if (!checkStrides(input->desc(), output->strides(), input->requiredStrides())) {
            needCopy = true;
            optionalCopy = false;
        }

        //
        // Check consumers StridesRequirement.
        //

        if (!needCopy) {
            for (const auto& consumerEdge : input->consumerEdges()) {
                const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

                if (consumerInfo.hasInput(consumerEdge)) {
                    const auto& consumerStrideReqs = consumerInfo.getInput(consumerEdge);
                    IE_ASSERT(input->checkStrides(consumerStrideReqs));

                    if (!checkStrides(input->desc(), output->strides(), consumerStrideReqs)) {
                        needCopy = true;
                        optionalCopy = false;
                    }
                }
            }
        }

        //
        // Check producer StridesRequirement.
        //

        if (!needCopy) {
            if (auto producerEdge = input->producerEdge()) {
                const auto& producerInfo = producerEdge->producer()->getDataStridesRequirements();

                if (producerInfo.hasOutput(producerEdge)) {
                    const auto& producerStrideReqs = producerInfo.getOutput(producerEdge);
                    IE_ASSERT(input->checkStrides(producerStrideReqs));

                    if (!checkStrides(input->desc(), output->strides(), producerStrideReqs)) {
                        needCopy = true;
                        optionalCopy = false;
                    }
                }

                if (!needCopy) {
                    //
                    // To reduce the size of HW output (still can be optimized).
                    //

                    if (producerEdge->producer()->category() == StageCategory::HW) {
                        needCopy = true;
                        optionalCopy = true;
                    }
                }
            }
        }
    }

    //
    // Insert Copy if needed
    //

    if (needCopy) {
        Data inputCopy;
        if (input->usage() == DataUsage::Const) {
            inputCopy = model->addNewData(
                input->name() + "@copy",
                input->desc());
        } else {
            inputCopy = model->duplicateData(
                input,
                "@copy");
            inputCopy->resetRequiredStrides();
        }

        auto copyStage = _stageBuilder->addCopyStage(
            model,
            formatString("%s@copy-for-expand", stage->name()),
            stage->origLayer(),
            input,
            inputCopy,
            "special::expand");
        copyStage->attrs().set<bool>("optional", optionalCopy);
        if (stage->attrs().has("batchInd")) {
            copyStage->attrs().set("batchInd", stage->attrs().get<int>("batchInd"));
        }

        model->replaceStageInput(stage->inputEdge(0), inputCopy);

        input = inputCopy;
    }

    //
    // Add Data<->Data edge
    //

    model->connectDatas()
        .parent(output)
        .child(input)
        .mode(SharedDataMode::ROI)
        .order(SharedDataOrder::ChildWritesToParent)
        .offset(offset)
        .done();
}

void SpecialStageProcessor::processCrop(
        const Model& model,
        const Stage& stage) {
    auto input = stage->input(0);
    auto output = stage->output(0);

    const auto& offset = stage->attrs().get<DimValues>("offset");

    IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());

    IE_ASSERT(offset.size() <= checked_cast<size_t>(input->desc().numDims()));
    for (const auto& p : offset) {
        IE_ASSERT(input->desc().dimsOrder().hasDim(p.first));
        IE_ASSERT(p.second + output->desc().dim(p.first) <= input->desc().dim(p.first));
    }

    //
    // Check if we need to insert Copy for output
    //

    bool needCopy = false;
    if (output->usage() != DataUsage::Intermediate) {
        needCopy = true;
    } else if (output->parentDataEdge() != nullptr) {
        needCopy = true;
    } else {
        //
        // Check output StridesRequirement.
        //

        IE_ASSERT(output->checkStrides(output->requiredStrides()));
        if (!checkStrides(output->desc(), input->strides(), output->requiredStrides())) {
            needCopy = true;
        }

        //
        // Check consumers StridesRequirement.
        //

        if (!needCopy) {
            for (const auto& consumerEdge : output->consumerEdges()) {
                const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

                if (consumerInfo.hasInput(consumerEdge)) {
                    const auto& consumerStrideReqs = consumerInfo.getInput(consumerEdge);
                    IE_ASSERT(output->checkStrides(consumerStrideReqs));

                    if (!checkStrides(output->desc(), input->strides(), consumerStrideReqs)) {
                        needCopy = true;
                        break;
                    }
                }
            }
        }
    }

    //
    // Insert output Copy if needed
    //

    if (needCopy) {
        auto outputCopy = model->duplicateData(
            output,
            "@copy");
        outputCopy->resetRequiredStrides();

        model->replaceStageOutput(stage->outputEdge(0), outputCopy);

        auto copyStage = _stageBuilder->addCopyStage(
            model,
            formatString("%s@copy-output-for-crop", stage->name()),
            stage->origLayer(),
            outputCopy,
            output,
            "special::crop");
        if (stage->attrs().has("batchInd")) {
            copyStage->attrs().set("batchInd", stage->attrs().get<int>("batchInd"));
        }

        output = outputCopy;
    }

    //
    // Add Data<->Data edge
    //

    model->connectDatas()
        .parent(input)
        .child(output)
        .mode(SharedDataMode::ROI)
        .order(SharedDataOrder::ParentWritesToChild)
        .offset(offset)
        .done();
}

void SpecialStageProcessor::processLoopStart(const Model& model, const Stage& stage) {
    for (const auto& input : stage->inputs()) {
        if (input->attrs().has("start-shared-allocation")) {
            const auto& src = input;
            const auto& dst = input->attrs().get<Data>("start-shared-allocation");

            VPU_THROW_UNLESS(src->canHaveAParent() || dst->canHaveAParent(), "for all back-edge connections required copy stages must be already introduced");

            auto parent = dst;
            auto child  = src;
            auto order  = SharedDataOrder::ChildWritesToParent;
            if (!src->canHaveAParent()) {
                std::swap(parent, child);
                order = SharedDataOrder::ParentWritesToChild;
            }

            model->connectDatas()
                .parent(parent)
                .child(child)
                .mode(SharedDataMode::ROI)
                .order(order)
                .connectionMode(SharedConnectionMode::SUBGRAPH)
                .done();
        }
    }

    for (const auto& backedge : stage->attrs().getOrDefault<HandleMultiMap<DataNode, Data>>("backedges", {})) {
        const auto& src = backedge.first;
        const auto& dst = backedge.second;

        // Tensor Iterator's body output data object must be a parent since it's not processed yet and don't have neither parent or child
        model->connectDatas()
            .parent(dst)
            .child(src)
            .mode(SharedDataMode::ROI)
            .order(SharedDataOrder::ChildWritesToParent)
            .connectionMode(SharedConnectionMode::SUBGRAPH)
            .done();
    }
}

void SpecialStageProcessor::processLoopEnd(const Model& model, const Stage& stage) {
    for (const auto& output : stage->outputs()) {
        if (output->attrs().has("end-shared-allocation")) {
            const auto& src = output->attrs().get<Data>("end-shared-allocation");
            const auto& dst = output;

            VPU_THROW_UNLESS(src->canHaveAParent() || dst->canHaveAParent(),
                "for all shared allocation connections required copy stages must be already introduced");

            auto parent = dst;
            auto child  = src;
            auto order  = SharedDataOrder::ChildWritesToParent;
            if (!src->canHaveAParent()) {
                std::swap(parent, child);
                order = SharedDataOrder::ParentWritesToChild;
            }

            model->connectDatas()
                .parent(parent)
                .child(child)
                .mode(SharedDataMode::ROI)
                .order(order)
                .connectionMode(SharedConnectionMode::SUBGRAPH)
                .done();
        }
    }
}

}  // namespace vpu
