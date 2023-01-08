// Copyright (C) 2018-2022 Intel Corporation
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

namespace vpu {

namespace {

class PassImpl : public Pass {
    // Nonintrusive stages do not modify values in tensor. Reorder is possible.
    struct StageUpliftSegment {
        Stage firstNonintrusiveStage;
        Stage lastNonintrusiveStage;
        Stage activationStage;
        void pushNonintrusiveStage(const Stage& stage) {
            firstNonintrusiveStage = stage;
            if (!lastNonintrusiveStage)
                lastNonintrusiveStage = stage;
        }
        bool empty() const {
            return !firstNonintrusiveStage || !lastNonintrusiveStage;
        }
    };
    using StageUpliftSegmentList = std::list<StageUpliftSegment>;

public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder)
        : _stageBuilder(stageBuilder) {
    }

    void run(const Model& model) override;

private:
    static StageUpliftSegmentList prepareStages(const Model& model) {
        StageUpliftSegmentList result;

        auto isApplicablePassThrough = [](const Stage& stage) {
            const auto type = stage->type();
            // We probably could do even more in future, but start with reliable ones.
            const bool isCopyType     = type == StageType::Copy;
            const bool isPermuteType  = type == StageType::Permute;
            const bool isReshapeType  = type == StageType::Reshape;
            const bool hasOneInput    = stage->numInputs() == 1;
            const bool hasOneOutput   = stage->numOutputs() == 1;
            const bool hasOneConsumer = stage->output(0)->numConsumers() == 1;
            return (isCopyType || isPermuteType || isReshapeType)
                    && hasOneInput
                    && hasOneOutput
                    && hasOneConsumer;
        };

        auto isApplicableActivation = [](const Stage& stage) {
            const auto type = stage->type();
            // @todo: also some eltwise could share this logic too: Clamp, Log, Exp, Floor, Pow(er?), etc.
            // @note: PRelu is not here because it is shape-dependent, which is dangerous for bypassing Permute.
            const bool isRelu    = type == StageType::Relu
                                || type == StageType::LeakyRelu
                                || type == StageType::BiasRelu
                                || type == StageType::BiasLeakyRelu;
            const bool isSigmoid = type == StageType::Sigmoid;
            const bool isTanh    = type == StageType::Tanh;
            const bool isErf     = type == StageType::Erf;

            return isRelu || isSigmoid || isTanh || isErf;
        };

        for (Stage stage : model->getStages()) {
            if (!isApplicableActivation(stage))
                continue;

            StageUpliftSegment segment;
            segment.activationStage = stage;

            while (true) {
                stage = stage->input(0)->producer();
                if (!stage || !isApplicablePassThrough(stage))
                    break;

                segment.pushNonintrusiveStage(stage);
            }

            if (!segment.empty())
                result.push_back(segment);
        }
        return result;
    }

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(upliftActivationStages);
    const auto stageUpliftPairList = prepareStages(model);

    for (const auto& stageUpliftPair : stageUpliftPairList) {
        Stage firstNIStage = stageUpliftPair.firstNonintrusiveStage;
        Stage lastNIStage  = stageUpliftPair.lastNonintrusiveStage;
        Stage actStage     = stageUpliftPair.activationStage;

       //  [oldInput]                     -(firstNIStage)-[firstOut]-..-[lastIn]-(lastNIStage)-[lastOut]-(ACT)-[actOut] - ..
       //  [oldInput]-(ACT)-[oldInputCopy]-(firstNIStage)-[firstOut]-..-[lastIn]-(lastNIStage)-               -[actOut] - ..

       // operations:
       //   oldInputCopy = duplicate of oldInput
       //   on ACT-input           : lastOut  => oldInput
       //   on ACT-output          : actOut   => oldInputCopy
       //   on firstNIStage-input  : oldInput => oldInputCopy
       //   on lastNIStage -output : lastOut  => actOut
       //   lastOut => removeUnused;

        Data oldInput     = firstNIStage->input(0);
        Data oldInputCopy = model->duplicateData(oldInput, "@relu-uplifting");
        Data actOut       = actStage->output(0);
        Data lastOut      = actStage->input(0);

        model->replaceStageInput (actStage->inputEdge(0)     , oldInput    );  // NOLINT
        model->replaceStageOutput(actStage->outputEdge(0)    , oldInputCopy);
        model->replaceStageInput (firstNIStage->inputEdge(0) , oldInputCopy);  // NOLINT
        model->replaceStageOutput(lastNIStage->outputEdge(0) , actOut      );  // NOLINT

        model->removeUnusedData(lastOut);
    }
}

}  // namespace

Pass::Ptr PassManager::upliftActivationStages() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
