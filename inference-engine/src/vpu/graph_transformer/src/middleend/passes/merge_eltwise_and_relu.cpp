// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <set>
#include <memory>

#include <vpu/middleend/sw/utility.hpp>

#include <vpu/configuration/options/enable_early_eltwise_relu_fusion.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

enum class MergeMode {
    DYNAMIC_NETWORK,
    STATIC_NETWORK
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(MergeMode mode) : m_mode(mode) {}
    void run(const Model& model) override;

private:
    MergeMode m_mode;
};

void PassImpl::run(const Model& model) {
    const bool enableEarlyEltwiseReLUFusion = CompileEnv::get().config.get<EnableEarlyEltwiseReluFusionOption>();
    if (enableEarlyEltwiseReLUFusion) {
        if (m_mode == MergeMode::DYNAMIC_NETWORK) {
            VPU_PROFILE(mergeEltwiseAndReLUDynamic);
            if (model->isStatic()) {
                return;
            }
        } else if (m_mode == MergeMode::STATIC_NETWORK) {
            VPU_PROFILE(mergeEltwiseAndReLUStatic);
            if (model->isDynamic()) {
                return;
            }
        }
    }

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

            const auto stridesAreSupported = reluInput->strides() == reluOutput->strides() || reluOutput->checkStrides(StridesRequirement::compact());
            if ((enableEarlyEltwiseReLUFusion && (stridesAreSupported || model->isDynamic())) || (!enableEarlyEltwiseReLUFusion && stridesAreSupported)) {
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

Pass::Ptr PassManager::mergeEltwiseAndReLUStatic() {
    return std::make_shared<PassImpl>(MergeMode::STATIC_NETWORK);
}

Pass::Ptr PassManager::mergeEltwiseAndReLUDynamic() {
    return std::make_shared<PassImpl>(MergeMode::DYNAMIC_NETWORK);
}

}  // namespace vpu
