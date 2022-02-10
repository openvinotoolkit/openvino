// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/special_stage_processor.hpp"

#include <vector>
#include <utility>

namespace vpu {

namespace {

struct NeedCopyDesc {
    bool isCopyNeed = false;
    bool isCopyOptional = false;
};

NeedCopyDesc isOutputCopyRequired(
                const Stage& stage,
                const StageOutput& outputEdge,
                const Data& inputData) {
    NeedCopyDesc needCopyDesc;
    auto output = outputEdge->output();
    if (output->usage() != DataUsage::Intermediate) {
        needCopyDesc.isCopyNeed = true;
    } else if (output->parentDataToDataEdge() != nullptr) {
        needCopyDesc.isCopyNeed = true;
    } else {
        //
        // Check output StridesRequirement
        //

        IE_ASSERT(output->checkStrides(output->requiredStrides()));
        if (!checkStrides(output->desc(), inputData->strides(), output->requiredStrides())) {
            needCopyDesc.isCopyNeed = true;
        }

        //
        // Check consumers StridesRequirement.
        //

        if (!needCopyDesc.isCopyNeed) {
            for (const auto& consumerEdge : output->consumerEdges()) {
                const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();
                if (consumerInfo.hasInput(consumerEdge)) {
                    const auto& consumerStrideReqs = consumerInfo.getInput(consumerEdge);
                    IE_ASSERT(output->checkStrides(consumerStrideReqs));
                    if (!checkStrides(output->desc(), inputData->strides(), consumerStrideReqs)) {
                        needCopyDesc.isCopyNeed = true;
                        break;
                    }
                }
            }
        }
    }
    return needCopyDesc;
}

NeedCopyDesc isInputCopyRequired(
                const Stage& stage,
                const StageInput& inputEdge,
                const Data& outputData) {
    auto input = inputEdge->input();
    NeedCopyDesc needCopyDesc;
    if (input->usage() != DataUsage::Intermediate) {
        needCopyDesc.isCopyNeed = true;
    } else if (input->parentDataToDataEdge() != nullptr) {
        needCopyDesc.isCopyNeed = true;
    } else {
        //
        // Check input StridesRequirement.
        //

        IE_ASSERT(input->checkStrides(input->requiredStrides()));
        if (!checkStrides(input->desc(), outputData->strides(), input->requiredStrides())) {
            needCopyDesc.isCopyNeed = true;
        }

        //
        // Check consumers StridesRequirement.
        //

        if (!needCopyDesc.isCopyNeed) {
            for (const auto& consumerEdge : input->consumerEdges()) {
                const auto& consumerInfo = consumerEdge->consumer()->getDataStridesRequirements();

                if (consumerInfo.hasInput(consumerEdge)) {
                    const auto& consumerStrideReqs = consumerInfo.getInput(consumerEdge);
                    IE_ASSERT(input->checkStrides(consumerStrideReqs));

                    if (!checkStrides(input->desc(), outputData->strides(), consumerStrideReqs)) {
                        needCopyDesc.isCopyNeed = true;
                    }
                }
            }
        }

        //
        // Check producer StridesRequirement.
        //

        if (!needCopyDesc.isCopyNeed) {
            if (auto producerEdge = input->producerEdge()) {
                const auto& producerInfo = producerEdge->producer()->getDataStridesRequirements();

                if (producerInfo.hasOutput(producerEdge)) {
                    const auto& producerStrideReqs = producerInfo.getOutput(producerEdge);
                    IE_ASSERT(input->checkStrides(producerStrideReqs));

                    if (!checkStrides(input->desc(), outputData->strides(), producerStrideReqs)) {
                        needCopyDesc.isCopyNeed = true;
                    }
                }

                if (!needCopyDesc.isCopyNeed) {
                    //
                    // To reduce the size of HW output (still can be optimized).
                    //

                    if (producerEdge->producer()->category() == StageCategory::HW) {
                        needCopyDesc.isCopyNeed = true;
                        needCopyDesc.isCopyOptional = true;
                    }
                }
            }
        }
    }

    return needCopyDesc;
}

Data insertCopyOfInput(const Model& model,
                       const Stage& stage,
                       const StageInput& edge,
                       const StageBuilder::Ptr& _stageBuilder,
                       const NeedCopyDesc& desc) {
    auto data = edge->input();

    Data copy;
    if (data->usage() == DataUsage::Const) {
        copy = model->addNewData(data->name() + "@copy", data->desc());
    } else {
        copy = model->duplicateData(data, "@copy");
        copy->resetRequiredStrides();
    }
    if (stage->type() == StageType::Reshape)
        copy->updateRequiredStrides(StridesRequirement::compact());

    bool hasMultipleInputs = stage->numInputs() > 1;
    auto inputNumStr = hasMultipleInputs ? formatString("@input=%d", edge->portInd()) : "";
    std::stringstream typeAsString;
    typeAsString << stage->type();

    auto copyStage = _stageBuilder->addCopyStage(
            model,
            formatString("%s%s@copy-for-%s", stage->name(), inputNumStr, typeAsString),
            stage->origLayer(),
            data,
            copy,
            formatString("special::%s", typeAsString));
    if (stage->type() != StageType::Reshape) {
        copyStage->attrs().set<bool>("optional", desc.isCopyOptional);
    }
    if (stage->attrs().has("batchInd")) {
        copyStage->attrs().set("batchInd", stage->attrs().get<int>("batchInd"));
    }

    model->replaceStageInput(edge, copy);

    return copy;
}

Data insertCopyOfOutput(const Model& model,
                        const Stage& stage,
                        const StageOutput& edge,
                        const StageBuilder::Ptr& _stageBuilder) {
    auto data = edge->output();
    auto copy = model->duplicateData(data, "@copy");
    copy->resetRequiredStrides();

    model->replaceStageOutput(edge, copy);

    bool hasMultipleOutputs = stage->numOutputs() > 1;
    auto outputNumStr = hasMultipleOutputs ? formatString("@output=%d", edge->portInd()) : "";
    std::stringstream typeAsString;
    typeAsString << stage->type();

    auto copyStage = _stageBuilder->addCopyStage(
            model,
            formatString("%s%s@copy-for-%s", stage->name(), outputNumStr, typeAsString),
            stage->origLayer(),
            copy,
            data,
            formatString("special::%s", typeAsString));
    if (stage->attrs().has("batchInd")) {
        copyStage->attrs().set("batchInd", stage->attrs().get<int>("batchInd"));
    }

    return copy;
}

} // namespace


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

        auto desc = isOutputCopyRequired(stage, outEdge, input);
        if (desc.isCopyNeed) {
            output = insertCopyOfOutput(model, stage, outEdge, _stageBuilder);
        }

        //
        // Add Data<->Data edge
        //

        model->connectDataWithData()
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

        NeedCopyDesc desc = isInputCopyRequired(stage, inEdge, output);
        if (desc.isCopyNeed) {
            input = insertCopyOfInput(model, stage, inEdge, _stageBuilder, desc);
        }

        //
        // Add Data<->Data edge
        //

        model->connectDataWithData()
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

    NeedCopyDesc desc;
    if ((input->usage() != DataUsage::Intermediate || input->parentDataToDataEdge() != nullptr) &&
        (output->usage() != DataUsage::Intermediate || output->parentDataToDataEdge() != nullptr))
        desc.isCopyNeed = true;
    if (desc.isCopyNeed) {
        input = insertCopyOfInput(model, stage, stage->inputEdge(0), _stageBuilder, desc);
    }

    //
    // Add Data<->Data edge
    //

    if (input->usage() == DataUsage::Intermediate &&
        input->parentDataToDataEdge() == nullptr) {
        model->connectDataWithData()
            .parent(output)
            .child(input)
            .mode(SharedDataMode::Reshape)
            .order(SharedDataOrder::ChildWritesToParent)
            .done();
    } else if (output->usage() == DataUsage::Intermediate &&
               output->parentDataToDataEdge() == nullptr) {
        model->connectDataWithData()
            .parent(input)
            .child(output)
            .mode(SharedDataMode::Reshape)
            .order(SharedDataOrder::ParentWritesToChild)
            .done();
    } else {
        IE_ASSERT(input->usage() == DataUsage::Intermediate &&
                  input->parentDataToDataEdge() == nullptr);
        IE_ASSERT(output->usage() == DataUsage::Intermediate &&
                  output->parentDataToDataEdge() == nullptr);
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

    auto desc = isInputCopyRequired(stage, stage->inputEdge(0), output);
    if (desc.isCopyNeed) {
        input = insertCopyOfInput(model, stage, stage->inputEdge(0), _stageBuilder, desc);
    }

    //
    // Add Data<->Data edge
    //

    model->connectDataWithData()
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

    auto desc = isOutputCopyRequired(stage, stage->outputEdge(0), input);
    if (desc.isCopyNeed) {
        output = insertCopyOfOutput(model, stage, stage->outputEdge(0), _stageBuilder);
    }

    model->connectDataWithData()
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

            model->connectDataWithData()
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
        model->connectDataWithData()
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

            model->connectDataWithData()
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
