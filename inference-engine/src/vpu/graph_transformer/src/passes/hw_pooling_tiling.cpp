// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <cmath>

#include <tuple>
#include <list>
#include <string>
#include <limits>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/passes/hw_conv_tiling/hw_convolution_tiler.hpp>
#include <vpu/passes/hw_pooling_tiling/hw_pooling_tiler.hpp>
#include <vpu/passes/hw_pooling_tiling/hw_stage_tiler.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuidler) : _stageBuilder(stageBuidler) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
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

        const HWPoolStageOptions so(origStage);
        const HWPoolStageIO sio(origStage, origStage->output(0));

        //
        // Try to find "best" tiling
        //

        const size_t tilingsCount = 1;
        const HWTilingNS::Direction direction =
                HWTilingNS::Direction::INPUT_TO_OUTPUT;
        // HWTilingNS::Direction::OUTPUT_TO_INPUT;

        const HWTilingNS::HWPoolingTiler tiler(
                HWTilingNS::ConvolutionOptions(origStage->name(),
                     sio.origInput->desc().dims(), sio.origOutput->desc().dims(),
                     sio.origOutput->desc().dims(),
                     so.kernelSizeX, so.kernelSizeY,
                     so.kernelStride,
                     so.padLeft, so.padRight, so.padTop, so.padBottom, false),
                direction, tilingsCount);

        if (!tiler.isTilingPossible()) {
            origStage->attrs().set<bool>("tryHW", false);

            auto swOutput = sio.origOutput;
            if (so.withReLU) {
                swOutput = model->addNewData(
                    origStage->name(),
                    sio.origOutput->desc());
                swOutput->attrs().copyFrom(sio.origOutput->attrs());

                model->replaceStageOutput(origStage->outputEdge(0), swOutput);

                _stageBuilder->addReLUStage(
                    model,
                    origStage->name() + "@ReLU",
                    origStage->origLayer(),
                    0.0,
                    swOutput,
                    sio.origOutput);
            }

            continue;
        }

        //
        // Create HW tiles
        //

        model->disconnectStage(origStage);


        for (const auto &tiling : tiler.getHwTilings()) {
            HWPoolStageTiler hwStageTiler(so, sio, model,
                                          origStage, _stageBuilder, tiling);
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
        //
        // Remove SW stage
        //

        model->removeStage(origStage);
    }
}

}  // namespace

Pass::Ptr PassManager::hwPoolTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
