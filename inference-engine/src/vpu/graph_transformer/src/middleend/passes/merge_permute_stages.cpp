// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <list>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/utils/error.hpp>

namespace vpu {

namespace {

const char permutationParamKey[] = "permutation";  // @todo: constant sharing with permute.cpp.
const char outputOrderKey[]      = "outputOrder";

class PassImpl : public Pass {
    struct StageMergeGroup {
        Stage first;
        std::list<Stage> merging;
        void push_back(const Stage& stage) {
            if (first)
                merging.push_back(stage);
            else
                first = stage;
        }
    };
    using StageMergeGroupList = std::list<StageMergeGroup>;

public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder)
        : _stageBuilder(stageBuilder) {
    }

    void run(const Model& model) override;

private:
    static PermutationIndexVector permuteVectorFromStageInternal(const Stage& permuteStage) {
        return permuteMapToVector(permuteStage->attrs().get<PermutationDimsMap>(permutationParamKey),
                                  permuteStage->input(0)->desc().dimsOrder(),
                                  permuteStage->output(0)->desc().dimsOrder());
    }

    static PermutationDimsMap permuteVectorToStageInternal(const PermutationIndexVector& permutation, const Stage& permuteStage) {
        return permuteVectorToMap(permutation,
                                  permuteStage->input(0)->desc().dimsOrder(),
                                  permuteStage->output(0)->desc().dimsOrder());
    }

    static bool isTrivialPermute(const PermutationIndexVector& permuteDims) {
        for (size_t i = 0; i < permuteDims.size(); ++i)
            if (i != permuteDims[i])
                return false;

        return true;
    }

    static StageMergeGroupList prepareStagesForMerge(const Model& model) {
        StageMergeGroupList result;
        StageMergeGroup buffer;
        StageSet visitedStages;
        auto consumeBuffer = [&buffer, &result]() {
            if (!buffer.merging.empty()) {
                result.push_back(buffer);
            }
            buffer = StageMergeGroup();
        };
        auto isApplicablePermute = [](const Stage& stage) {
            const bool isPermuteType = stage->type() == StageType::Permute;
            VPU_THROW_UNLESS(
                !isPermuteType || stage->attrs().has(permutationParamKey),
                "Invalid Permute Stage node %v: missing %v attribute",
                stage, permutationParamKey);
            return isPermuteType;
        };

        auto isApplicableSequentalPermute = [](const Stage& stage) {
            const auto stageOutput          = stage->output(0);
            const bool needToPreserveOutput = stageOutput->usage() == DataUsage::Output;
            const auto consumersCount       = stageOutput->consumers().size();
            return !needToPreserveOutput && consumersCount <= 1;
        };

        auto processStage = [&visitedStages, &isApplicablePermute, &isApplicableSequentalPermute, &buffer](const Stage& stage) -> Stage {
            if (visitedStages.find(stage) != visitedStages.cend())
                return nullptr;

            visitedStages.insert(stage);

            if (!isApplicablePermute(stage)) {
                return nullptr;
            }

            VPU_THROW_UNLESS(
                stage->numOutputs() == 1,
                "Unexpected Permute Stage node %v, too many outputs : %v",
                stage, stage->numOutputs());

            buffer.push_back(stage);

            if (!isApplicableSequentalPermute(stage)) {
                return nullptr;
            }

            auto permuteConsumers = stage->output(0)->consumers();
            return permuteConsumers.empty() ? nullptr : permuteConsumers.front();
        };

        for (Stage stage : model->getStages()) {
            do {
                stage = processStage(stage);
            } while (stage != nullptr);

            consumeBuffer();
        }
        return result;
    }

 private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(mergePermuteStages);
    const StageMergeGroupList stageMergeGroupList = prepareStagesForMerge(model);

    for (const auto& stageMergeGroup : stageMergeGroupList) {
        const auto& firstPermuteStage = stageMergeGroup.first;
        auto resultPermutation = permuteVectorFromStageInternal(firstPermuteStage);

        DimsOrder outputLayout;

        // remove merging stages from model, gluing corresponing inputs.
        for (const auto& permuteStage : stageMergeGroup.merging) {
            auto permuteInput  = permuteStage->input(0);
            auto permuteOutput = permuteStage->output(0);

            outputLayout = permuteStage->attrs().getOrDefault(outputOrderKey, outputLayout);

            const auto permutation = permuteVectorFromStageInternal(permuteStage);
            resultPermutation = combinePermutationVectors(resultPermutation, permutation);

            model->removeStage(permuteStage);
            model->replaceStageOutput(firstPermuteStage->outputEdge(0), permuteOutput);

            model->removeUnusedData(permuteInput);
        }
        firstPermuteStage->attrs().set(permutationParamKey, permuteVectorToStageInternal(resultPermutation, firstPermuteStage));
        if (!outputLayout.empty())
            firstPermuteStage->attrs().set(outputOrderKey, outputLayout);

        // if we have no actual permutation, replace it with copy.
        if (isTrivialPermute(resultPermutation)) {
            auto permuteInput  = firstPermuteStage->input(0);
            auto permuteOutput = firstPermuteStage->output(0);
            if (permuteInput->desc().dimsOrder() == permuteOutput->desc().dimsOrder()) {
                auto stageName     = firstPermuteStage->name();
                auto origLayer     = firstPermuteStage->origLayer();
                model->removeStage(firstPermuteStage);

                auto copyStage = _stageBuilder->addCopyStage(model, stageName + "@merged-to-copy",
                                                             origLayer, permuteInput, permuteOutput, "Eliminated permute");
                // TODO: make this optional=true with corresponding fixes in eliminate_copy (it expects Special stages now).
                copyStage->attrs().set("optional", false);
            }
        }
    }
}

}  // namespace

Pass::Ptr PassManager::mergePermuteStages() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
