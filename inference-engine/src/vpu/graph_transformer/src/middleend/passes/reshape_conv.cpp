// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This pass changes geometry of convolution stages in order
// to get more efficient HW tiling (pass "hwConvTiling") using reshape stages.

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/middleend/hw/conv_tiling/reshape_conv_func.hpp>

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
    VPU_PROFILE(hwConvTiling);
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

        std::string name = stage->name();
        int inputC = inputDesc.dim(Dim::C);
        int outputC = outputDesc.dim(Dim::C);
        int dimH = inputDesc.dim(Dim::H);
        int dimW = inputDesc.dim(Dim::W);
        int resultW = 0, resultH = 0;
        resultH = ChoiceDimH(name, inputC, outputC, dimH, dimW);

        if (stage->origLayer()->params.count("ConvReshape")) {
            std::string rtParam = stage->origLayer()->params.at("ConvReshape");
            try {
                resultW = std::stoi(rtParam);
                resultH = dimH * dimW / resultW;
            } catch (...) {
                resultW = resultH = 0;
            }
        }

        if (resultH == 0) {
            continue;
        }
        resultW = dimH * dimW / resultH;

        auto newDesc_input = inputDesc;
        newDesc_input.setDim(Dim::W, resultW);
        newDesc_input.setDim(Dim::H, resultH);

        auto newDesc_output = outputDesc;
        newDesc_output.setDim(Dim::W, resultW);
        newDesc_output.setDim(Dim::H, resultH);

        auto newInput = model->duplicateData(input, "@input-data-after-reshape",
                newDesc_input);

        auto newOutput = model->duplicateData(output, "@output-data-before-reshape",
                newDesc_output);

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
