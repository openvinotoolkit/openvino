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
    VPU_PROFILE(removeUnusedStagesOutputs);

    for (const auto& stage : model->getStages()) {
        if (stage == nullptr || ((stage->type() != StageType::LSTMCell) && (stage->type() != StageType::TopK))) {
            continue;
        }

        for (const auto& outEdge : stage->outputEdges()) {
            auto output = outEdge->output();

            if (output->usage() == DataUsage::Intermediate && output->numConsumers() == 0) {
                _stageBuilder->addNoneStage(model, stage->name() + "@fake-none", stage->origLayer(), {output}, {});
            }
        }
    }
}

}  // namespace

Pass::Ptr PassManager::removeUnusedStagesOutputs() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
