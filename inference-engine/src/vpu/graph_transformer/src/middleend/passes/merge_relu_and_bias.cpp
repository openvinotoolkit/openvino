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
    VPU_PROFILE(mergeReLUAndBias);

    for (const auto& biasStage : model->getStages()) {
        if (biasStage == nullptr) {
            continue;
        }

        if (biasStage->type() != StageType::Bias) {
            continue;
        }

        if (auto reluStage = getOneOfSingleNextStage(biasStage, {StageType::Relu, StageType::LeakyRelu})) {
            auto biasInput = biasStage->input(0);
            auto biases = biasStage->input(1);

            auto reluOutput = reluStage->output(0);

            auto reluStageName = reluStage->name();
            auto reluOrigLayer = reluStage->origLayer();
            auto negativeSlope = reluStage->attrs().get<float>("negativeSlope");

            model->removeStage(biasStage);
            model->removeStage(reluStage);

            _stageBuilder->addReLUStage(
                model,
                reluStageName,
                reluOrigLayer,
                negativeSlope,
                biasInput,
                reluOutput,
                biases);
        }
    }
}

}  // namespace

Pass::Ptr PassManager::mergeReLUAndBias() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
