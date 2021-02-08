// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace vpu {

namespace {

class PassImpl final : public PerStagePass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : PerStagePass({StageType::Convert}),
                                                        _stageBuilder(std::move(stageBuilder)) {}

    void runForStage(const Model& model, const Stage& convert) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::runForStage(const Model& model, const Stage& convert) {
    VPU_PROFILE(eliminateRedundantConversions);

    const auto input = convert->input(0);
    const auto output = convert->output(0);

    //
    // Check and remove the convert that was added to unused input
    // In this case we will have the converted intermediate data object which is not consumed
    //

    if (output->usage() == DataUsage::Intermediate && !output->isConsumed()) {
        model->removeStage(convert);
        model->removeUnusedData(output);
        return;
    }

    //
    // We remove Convert stage if input and output data types are equal.
    // It could happen if there was a non-IO FP16 <-> FP32 conversion in
    // original net and GraphTransformer's frontend had changed FP32 to FP16
    // when parsed layers' inputs and outputs.
    //

    if (input->desc().type() != output->desc().type()) {
        return;
    }

    model->disconnectStage(convert);

    if (output->usage() == DataUsage::Output) {
        _stageBuilder->addCopyStage(
                model,
                convert->name() + "@replaced-with-copy",
                convert->origLayer(),
                input,
                output,
                "eliminateRedundantConversions");
    } else {
        VPU_INTERNAL_CHECK(output->numConsumers() > 0,
                           "eliminateRedundantConversions: Convert stage with name %s "
                           "has no consumers", convert->name());
        for (const auto consumerEdge : output->consumerEdges()) {
            model->replaceStageInput(consumerEdge, input);
        }
        VPU_INTERNAL_CHECK(output->numConsumers() == 0,
                           "eliminateRedundantConversions: Convert stage with name %s "
                           "must have no consumers after elimination", convert->name());
    }

    model->removeStage(convert);
}

}  // namespace

Pass::Ptr PassManager::eliminateRedundantConversions() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
