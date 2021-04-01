// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <set>
#include <list>
#include <string>
#include <utility>
#include <memory>

#include <vpu/stages/stub_stage.hpp>
#include <vpu/middleend/hw/conv_tiling/hw_convolution_tiler.hpp>
#include <vpu/middleend/hw/pooling_tiling/hw_pooling_tiler.hpp>
#include <vpu/middleend/hw/pooling_tiling/hw_stage_tiler.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(hwPoolTiling);

    for (const auto& origStage : model->getStages()) {
        if (origStage->type() != StageType::StubMaxPool &&
            origStage->type() != StageType::StubAvgPool) {
            continue;
        }

        auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        const HWPoolStageOptions stageOptions(origStage);
        const HWPoolStageIO stageIO(origStage, origStage->output(0));

        //
        // Try to find "best" tiling
        //

        const size_t tilingsCount = 1;
        const HWTilingNS::Direction direction =
                HWTilingNS::Direction::INPUT_TO_OUTPUT;
        // HWTilingNS::Direction::OUTPUT_TO_INPUT;

        const auto convolutionOptions = HWTilingNS::ConvolutionOptions{
            origStage->name(),
            stageIO.origInput->desc().dims(),
            stageIO.origOutput->desc().dims(),
            stageIO.origOutput->desc().dims(),
            stageOptions.kernelSizeX,
            stageOptions.kernelSizeY,
            stageOptions.kernelStride,
            stageOptions.padLeft,
            stageOptions.padRight,
            stageOptions.padTop,
            stageOptions.padBottom,
            false};

        const HWTilingNS::HWPoolingTiler tiler(convolutionOptions, direction, tilingsCount);

        if (!tiler.isTilingPossible()) {
            origStage->attrs().set<bool>("tryHW", false);

            auto swOutput = stageIO.origOutput;
            if (stageOptions.withReLU) {
                swOutput = model->addNewData(origStage->name(), stageIO.origOutput->desc());
                swOutput->attrs().copyFrom(stageIO.origOutput->attrs());

                model->replaceStageOutput(origStage->outputEdge(0), swOutput);

                _stageBuilder->addReLUStage(
                    model,
                    origStage->name() + "@ReLU",
                    origStage->origLayer(),
                    0.0,
                    swOutput,
                    stageIO.origOutput);
            }

            continue;
        }

        //
        // Create HW tiles
        //

        model->disconnectStage(origStage);


        for (const auto &tiling : tiler.getHwTilings()) {
            HWPoolStageTiler hwStageTiler(stageOptions, stageIO, model, origStage, _stageBuilder, tiling);
            //
            // Split/concat input/output tiles
            //

            if (!hwStageTiler.hwInputTiles.empty()) {
                _stageBuilder->addSplitStage(
                    model,
                    origStage->name() + "@split-input",
                    origStage->origLayer(),
                    std::move(hwStageTiler.hwInputTilesOffsets),
                    hwStageTiler.hwInput,
                    hwStageTiler.hwInputTiles);
            }

            if (!hwStageTiler.hwOutputTiles.empty()) {
                _stageBuilder->addConcatStage(
                    model,
                    origStage->name() + "@concat-output",
                    origStage->origLayer(),
                    std::move(hwStageTiler.hwOutputTilesOffsets),
                    hwStageTiler.hwOutputTiles,
                    hwStageTiler.hwOutput);
            }
        }

        model->removeStage(origStage);
    }
}

}  // namespace

Pass::Ptr PassManager::hwPoolTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
