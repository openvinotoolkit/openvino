// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This pass changes geometry of convolution stages in order
// to get more efficient HW tiling (pass "hwConvTiling") using reshape stages.

#include <vpu/middleend/pass_manager.hpp>

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
    VPU_PROFILE(reshapeBeforeConvTiling);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubConv) {
            continue;
        }

        const auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        const auto input = stage->input(0);
        const auto output = stage->output(0);

        const auto& inputDesc = input->desc();
        const auto& outputDesc = output->desc();

        if ((inputDesc.dimsOrder() != DimsOrder::NCHW) ||
            (outputDesc.dimsOrder() != DimsOrder::NCHW)) {
            continue;
        }

        if (stage->attrs().get<int>("kernelSizeX") != 1 ||
            stage->attrs().get<int>("kernelSizeY") != 1)
            continue;

        int dimH = inputDesc.dim(Dim::H);
        int dimW = inputDesc.dim(Dim::W);
        int resultH = 0;
        int resultW = 0;

        // need to invistigate
        // if (stage->origLayer()->params.count("alt_width")) {
        //     const auto alt_width = stage->origLayer()->params.at("alt_width");
        //     if (!alt_width.empty() &&
        //         std::find_if(alt_width.begin(), alt_width.end(),
        //                      [](unsigned char c) { return !std::isdigit(c); }) == alt_width.end()) {
        //         resultW = std::stoul(alt_width);
        //     }
        // }

        if (resultW == 0) {
            continue;
        }

        resultH = dimH * dimW / resultW;

        IE_ASSERT(dimH * dimW == resultH * resultW);

        auto inputNewDesc = inputDesc;
        inputNewDesc.setDim(Dim::W, resultW);
        inputNewDesc.setDim(Dim::H, resultH);

        auto outputNewDesc = outputDesc;
        outputNewDesc.setDim(Dim::W, resultW);
        outputNewDesc.setDim(Dim::H, resultH);

        auto newInput = model->duplicateData(input, "@input-data-after-reshape",
                inputNewDesc);

        auto newOutput = model->duplicateData(output, "@output-data-before-reshape",
                outputNewDesc);

        model->replaceStageInput(stage->inputEdge(0), newInput);
        model->replaceStageOutput(stage->outputEdge(0), newOutput);

        _stageBuilder->addReshapeStage(model,
                stage->name() + "@copy-reinterpret-input-data",
                nullptr, input, newInput);

        _stageBuilder->addReshapeStage(model,
                stage->name() + "@copy-reinterpret-output-data",
                nullptr, newOutput, output);
    }
}

}  // namespace

Pass::Ptr PassManager::reshapeBeforeConvTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
