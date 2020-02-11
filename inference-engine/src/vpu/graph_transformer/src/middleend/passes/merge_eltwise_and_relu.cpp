// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <set>
#include <memory>

#include <vpu/middleend/sw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(mergeEltwiseAndReLU);

    for (const auto& eltwiseStage : model->getStages()) {
        if (eltwiseStage == nullptr) {
            continue;
        }

        if (eltwiseStage->type() != StageType::Sum            &&
            eltwiseStage->type() != StageType::Prod           &&
            eltwiseStage->type() != StageType::Max            &&
            eltwiseStage->type() != StageType::Min            &&
            eltwiseStage->type() != StageType::Div            &&
            eltwiseStage->type() != StageType::Squared_diff   &&
            eltwiseStage->type() != StageType::Floor_mod      &&
            eltwiseStage->type() != StageType::Pow            &&
            eltwiseStage->type() != StageType::Equal          &&
            eltwiseStage->type() != StageType::Not_equal      &&
            eltwiseStage->type() != StageType::Less           &&
            eltwiseStage->type() != StageType::Less_equal     &&
            eltwiseStage->type() != StageType::Greater        &&
            eltwiseStage->type() != StageType::Greater_equal  &&
            eltwiseStage->type() != StageType::Logical_AND    &&
            eltwiseStage->type() != StageType::Logical_OR     &&
            eltwiseStage->type() != StageType::Logical_XOR    &&
            eltwiseStage->type() != StageType::Logical_NOT) {
            continue;
        }

        const bool allInputsAreFP16 = std::all_of(eltwiseStage->inputs().begin(), eltwiseStage->inputs().end(),
            [](const Data& data) { return data->desc().type() == DataType::FP16; });

        const bool allOutputsAreFP16 = std::all_of(eltwiseStage->outputs().begin(), eltwiseStage->outputs().end(),
            [](const Data& data) { return data->desc().type() == DataType::FP16; });

        if (!allInputsAreFP16 || !allOutputsAreFP16) {
            continue;
        }

        if (auto reluStage = getOneOfSingleNextStage(eltwiseStage, {StageType::Relu, StageType::LeakyRelu, StageType::Clamp})) {
            auto reluInput = reluStage->input(0);
            auto reluOutput = reluStage->output(0);

            if (reluInput->strides() == reluOutput->strides() || reluOutput->checkStrides(StridesRequirement::compact())) {
                auto reluStageType = reluStage->type();
                auto reluStageName = reluStage->name();

                auto negativeSlope = reluStage->attrs().getOrDefault<float>("negativeSlope", 0.0f);
                auto min_value = reluStage->attrs().getOrDefault<float>("min_value", 0.0f);
                auto max_value = reluStage->attrs().getOrDefault<float>("max_value", 1.0f);

                model->removeStage(reluStage);
                model->replaceStageOutput(eltwiseStage->outputEdge(0), reluOutput);

                auto namePostfix = " + " + reluStageName;
                eltwiseStage->appendNamePostfix(namePostfix);
                eltwiseStage->attrs().set<StageType>("postOperation", reluStageType);
                eltwiseStage->attrs().set<float>("negativeSlope", negativeSlope);
                eltwiseStage->attrs().set<float>("min_value", min_value);
                eltwiseStage->attrs().set<float>("max_value", max_value);
            }
        }
    }
}

}  // namespace

Pass::Ptr PassManager::mergeEltwiseAndReLU() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
