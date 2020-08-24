// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This is the pass that finds the pattern which consists of a Convolution stage
// with 2 consumers - Power and Concat stages (Power stage is also followed by Concat),
// ScaleShift (or Scale, it depends on biases) which comes after Concat,
// and the last one is Relu.

#include <vpu/middleend/pass_manager.hpp>

// #include <vpu/middleend/hw/conv_tiling/hw_stage_tiler.hpp>

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

        if (stage->attrs().get<int>("kernelSizeX") != 1 ||
            stage->attrs().get<int>("kernelSizeY") != 1) {
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

        const auto actualInputShape = inputDesc.dims();
        const auto expectedInputShape = DimValues{ {Dim::N, 1}, {Dim::C, 608}, {Dim::H, 34}, {Dim::W, 60} };

        if (actualInputShape != expectedInputShape) {
            continue;
        }

        if (outputDesc.dim(Dim::W) != inputDesc.dim(Dim::W) ||
            outputDesc.dim(Dim::H) != inputDesc.dim(Dim::H)) {
            continue;
        }

        std::map<size_t, size_t> outChannels{ {10, 1}, {128, 2}, {490, 3}};

        if (outChannels.find(outputDesc.dim(Dim::C)) == outChannels.end()) {
            continue;
        }

        auto newDesc_input = inputDesc;
        newDesc_input.setDim(Dim::W, 255);
        newDesc_input.setDim(Dim::H, 8);

        auto newDesc_output = outputDesc;
        newDesc_output.setDim(Dim::W, 255);
        newDesc_output.setDim(Dim::H, 8);

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
                stage->name() + "@copy-reinterpret-input-data",
                nullptr, newOutput, output);
    }
}

}  // namespace

Pass::Ptr PassManager::reshapeBeforeConvTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
