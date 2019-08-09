// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>

#include <vpu/allocator.hpp>
#include <vpu/utils/extra.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    void processConcat(const Model::Ptr& model, const Stage& stage);
    void processSplit(const Model::Ptr& model, const Stage& stage);
    void processReshape(const Model::Ptr& model, const Stage& stage);
    void processBroadcast(const Model::Ptr& model, const Stage& stage);
    void processShrink(const Model::Ptr& model, const Stage& stage);

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(processSpecialStages);

    //
    // Merge multiple Broadcast stages applied to the same input.
    //

    for (const auto& curBroadcastStage : model->getStages()) {
        if (curBroadcastStage == nullptr) {
            continue;
        }

        if (curBroadcastStage->type() != StageType::Broadcast) {
            continue;
        }

        auto input = curBroadcastStage->input(0);
        auto output = curBroadcastStage->output(0);

        bool hasDuplicates = false;
        for (const auto& inputConsumer : input->consumers()) {
            if (inputConsumer->type() != StageType::Broadcast) {
                continue;
            }

            if (inputConsumer == curBroadcastStage) {
                continue;
            }

            hasDuplicates = true;

            auto otherOutput = inputConsumer->output(0);

            if (otherOutput->desc().dims() != output->desc().dims()) {
                hasDuplicates = false;
                break;
            }

            if (otherOutput->usage() != DataUsage::Intermediate) {
                hasDuplicates = false;
                break;
            }
        }

        if (!hasDuplicates) {
            continue;
        }

        for (const auto& inputConsumer : input->consumers()) {
            if (inputConsumer->type() != StageType::Broadcast) {
                continue;
            }

            if (inputConsumer == curBroadcastStage) {
                continue;
            }

            auto otherOutput = inputConsumer->output(0);

            for (const auto& outputConsumerEdge : otherOutput->consumerEdges()) {
                model->replaceStageInput(outputConsumerEdge, output);
            }

            model->removeStage(inputConsumer);
        }
    }

    //
    // Add Copy stages when needed.
    //

    for (const auto& stage : model->getStages()) {
        if (stage == nullptr) {
            continue;
        }

        if (stage->type() == StageType::Concat) {
            processConcat(model, stage);
        } else if (stage->type() == StageType::Split) {
            processSplit(model, stage);
        } else if (stage->type() == StageType::Reshape) {
            processReshape(model, stage);
        } else if (stage->type() == StageType::Broadcast) {
            processBroadcast(model, stage);
        } else if (stage->type() == StageType::Shrink) {
            processShrink(model, stage);
        }
    }
}

void PassImpl::processConcat(const Model::Ptr& model, const Stage& stage) {
    auto output = stage->output(0);

    const auto& offsets = stage->attrs().get<std::vector<DimValues>>("offsets");
    IE_ASSERT(offsets.size() == stage->numInputs());

    for (const auto& inEdge : stage->inputEdges()) {
        IE_ASSERT(inEdge->portInd() >= 0);
        IE_ASSERT(inEdge->portInd() < offsets.size());

        auto input = inEdge->input();
        const auto& offsetFromOutput = offsets[inEdge->portInd()];

        IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());
        IE_ASSERT(offsetFromOutput.size() <= output->desc().numDims());
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
                inputCopy);
            copyStage->attrs().set<bool>("optional", optionalCopy);

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

void PassImpl::processSplit(const Model::Ptr& model, const Stage& stage) {
    auto input = stage->input(0);

    const auto& offsets = stage->attrs().get<std::vector<DimValues>>("offsets");
    IE_ASSERT(offsets.size() == stage->numOutputs());

    for (const auto& outEdge : stage->outputEdges()) {
        IE_ASSERT(outEdge->portInd() >= 0);
        IE_ASSERT(outEdge->portInd() < offsets.size());

        auto output = outEdge->output();
        const auto& offsetFromInput = offsets[outEdge->portInd()];

        IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());
        IE_ASSERT(offsetFromInput.size() <= input->desc().numDims());
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
            auto outputCopy = model->duplicateData(
                output,
                "@copy");
            outputCopy->resetRequiredStrides();

            auto outPortInd = outEdge->portInd();

            model->replaceStageOutput(outEdge, outputCopy);

            _stageBuilder->addCopyStage(
                model,
                formatString("%s@output=%d@copy-for-split", stage->name(), outPortInd),
                stage->origLayer(),
                outputCopy,
                output);

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

void PassImpl::processReshape(const Model::Ptr& model, const Stage& stage) {
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

        _stageBuilder->addCopyStage(
            model,
            formatString("%s@copy-for-reshape", stage->name()),
            stage->origLayer(),
            input,
            inputCopy);

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

void PassImpl::processBroadcast(const Model::Ptr& model, const Stage& stage) {
    auto input = stage->input(0);
    auto output = stage->output(0);

    const auto& offset = stage->attrs().get<DimValues>("offset");

    IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());

    IE_ASSERT(offset.size() <= output->desc().numDims());
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
            formatString("%s@copy-for-broadcast", stage->name()),
            stage->origLayer(),
            input,
            inputCopy);
        copyStage->attrs().set<bool>("optional", optionalCopy);

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

void PassImpl::processShrink(const Model::Ptr& model, const Stage& stage) {
    auto input = stage->input(0);
    auto output = stage->output(0);

    const auto& offset = stage->attrs().get<DimValues>("offset");

    IE_ASSERT(input->desc().dimsOrder() == output->desc().dimsOrder());

    IE_ASSERT(offset.size() <= input->desc().numDims());
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

        _stageBuilder->addCopyStage(
            model,
            formatString("%s@copy-output-for-shrink", stage->name()),
            stage->origLayer(),
            outputCopy,
            output);

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

}  // namespace

Pass::Ptr PassManager::processSpecialStages() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
