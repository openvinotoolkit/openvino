// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This is the pass that finds the pattern which consists of a Convolution stage
// with 2 consumers - Power and Concat stages (Power stage is also followed by Concat),
// ScaleShift (or Scale, it depends on biases) which comes after Concat,
// and the last one is Relu.

#include <vpu/middleend/pass_manager.hpp>

#include <set>
#include <memory>
#include <vector>

#include <vpu/middleend/sw/utility.hpp>
#include <vpu/model/data.hpp>

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
    VPU_PROFILE(replaceWithSCReLU);

    for (const auto& convolutionStage : model->getStages()) {
        if (convolutionStage == nullptr) {
            continue;
        }

        const auto nextStages = getExactNextStages(convolutionStage, {StageType::Power, StageType::StubConcat});
        if (nextStages.size() != 2 || convolutionStage->type() != StageType::StubConv) {
            continue;
        }

        auto& powerStage = nextStages.front();
        auto& concatStage = nextStages.back();

        const auto& biasPower = powerStage->attrs().get<float>("bias");
        const auto& scalePower = powerStage->attrs().get<float>("scale");
        const auto& powerPower = powerStage->attrs().get<float>("power");

        if (biasPower != 0.0f || scalePower != -1.0f || powerPower != 1.0f) {
            continue;
        }

        auto concatAfterPowerStage = getOneOfSingleNextStage(powerStage, {StageType::StubConcat});
        if (concatAfterPowerStage != concatStage) {
            continue;
        }

        auto scaleShiftStage = getOneOfSingleNextStage(concatStage, {StageType::ScaleShift, StageType::Scale, StageType::Bias});
        if (scaleShiftStage == nullptr) {
            continue;
        }

        auto reluStage = getOneOfSingleNextStage(scaleShiftStage, {StageType::Relu, StageType::LeakyRelu});
        if (reluStage == nullptr) {
            continue;
        }

        Data biases = nullptr;
        Data scales = nullptr;

        if (scaleShiftStage->type() == StageType::Bias) {
            biases = scaleShiftStage->input(1);
        }

        if (scaleShiftStage->type() == StageType::ScaleShift) {
            biases = scaleShiftStage->input(2);
        }

        if (scaleShiftStage->type() == StageType::ScaleShift || scaleShiftStage->type() == StageType::Scale) {
            scales = scaleShiftStage->input(1);
        }

        auto negSlope = reluStage->attrs().getOrDefault<float>("negativeSlope", 1.0f);
        auto axis = concatStage->attrs().getOrDefault<Dim>("axis", Dim::Invalid);
        if (axis == Dim::Invalid) {
            continue;
        }

        auto reluStageOut = reluStage->output(0);
        auto convStageOut = convolutionStage->output(0);

        model->removeStage(scaleShiftStage);
        model->removeStage(concatStage);
        model->removeStage(powerStage);
        model->removeStage(reluStage);

        _stageBuilder->addSCReluStage(
                model,
                "SCRelu",
                convolutionStage->origLayer(),
                negSlope,
                axis,
                convStageOut,
                reluStageOut,
                scales,
                biases);
    }
}

}  // namespace

Pass::Ptr PassManager::replaceWithSCReLU() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
