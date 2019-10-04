// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <set>
#include <memory>

#include <vpu/sw/utility.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(removeUnusedStagesOutputs);

    for (const auto& stage : model->getStages()) {
        if (stage == nullptr || ((stage->type() != StageType::LSTMCell) && (stage->type() != StageType::TopK))) {
            continue;
        }

        for (const auto& outEdge : stage->outputEdges()) {
            auto output = outEdge->output();

            if (output->usage() == DataUsage::Intermediate && output->numConsumers() == 0) {
                model->replaceStageOutput(outEdge, model->addFakeData());
            }
        }
    }
}

}  // namespace

Pass::Ptr PassManager::removeUnusedStagesOutputs() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
