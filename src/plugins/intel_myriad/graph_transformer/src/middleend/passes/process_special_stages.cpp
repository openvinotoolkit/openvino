// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>

#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/middleend/special_stage_processor.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) :
            _stageBuilder(stageBuilder), _processor(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
    SpecialStageProcessor _processor;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(processSpecialStages);

    //
    // Merge multiple Expand stages applied to the same input.
    //

    for (const auto& curExpandStage : model->getStages()) {
        if (curExpandStage == nullptr) {
            continue;
        }

        if (curExpandStage->type() != StageType::Expand) {
            continue;
        }

        auto input = curExpandStage->input(0);
        auto output = curExpandStage->output(0);

        bool hasDuplicates = false;
        for (const auto& inputConsumer : input->consumers()) {
            if (inputConsumer->type() != StageType::Expand) {
                continue;
            }

            if (inputConsumer == curExpandStage) {
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
            if (inputConsumer->type() != StageType::Expand) {
                continue;
            }

            if (inputConsumer == curExpandStage) {
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

        if (stage->type() == StageType::StubConcat) {
            _processor.processConcat(model, stage);
        } else if (stage->type() == StageType::Split) {
            _processor.processSplit(model, stage);
        } else if (stage->type() == StageType::Reshape) {
            _processor.processReshape(model, stage);
        } else if (stage->type() == StageType::Expand) {
            _processor.processExpand(model, stage);
        } else if (stage->type() == StageType::Crop) {
            _processor.processCrop(model, stage);
        } else if (stage->type() == StageType::LoopStart) {
            _processor.processLoopStart(model, stage);
        } else if (stage->type() == StageType::LoopEnd) {
            _processor.processLoopEnd(model, stage);
        }
    }
}

}  // namespace

Pass::Ptr PassManager::processSpecialStages() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
