// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <cmath>

#include <tuple>
#include <vector>
#include <limits>
#include <algorithm>
#include <list>
#include <string>
#include <memory>
#include <utility>
#include <set>
#include <array>

#include <precision_utils.h>

#include <vpu/compile_env.hpp>
#include <vpu/stub_stage.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/tiling.hpp>
#include <vpu/hw/utility.hpp>

namespace vpu {

namespace {

std::tuple<int, int, HwFullyConnectedTileInfo> splitFullyConnected(
        int inN, int outN,
        const SmallVector<HwOpMode>& modes = {HwOpMode::MODE_1_256, HwOpMode::MODE_2_128, HwOpMode::MODE_4_64, HwOpMode::MODE_8_32, HwOpMode::MODE_16_16}) {
    struct Solution final {
        HwOpMode mode = HwOpMode::MODE_1_256;
        int newInN = 0;
        int newOutN = 0;
        int workInN = 0;
        int workOutN = 0;
        int countIn = 0;
        int countOut = 0;
        int cost = std::numeric_limits<int>::max();
    };

    Solution bestSol;

    for (auto mode : modes) {
        auto ramBlocks = 1 << static_cast<int>(mode);
        auto maxInN = ramBlocks * 256;
        auto maxOutN = 256 / ramBlocks;
        auto newInN = alignVal(inN, ramBlocks);
        auto newOutN = alignVal(outN, 8);
        auto workInN = std::min(newInN, maxInN);
        auto workOutN = std::min(newOutN, maxOutN);

        if (workInN < ramBlocks) {
            continue;
        }

        auto countIn = static_cast<int>(std::ceil(static_cast<double>(newInN) / workInN));
        auto countOut = static_cast<int>(std::ceil(static_cast<double>(newOutN) / workOutN));
        auto cost = countIn * countOut * (workInN / ramBlocks + CNN_MODES_COST[static_cast<int>(mode)]);

        Solution curSol;
        curSol.mode = mode;
        curSol.newInN = newInN;
        curSol.newOutN = newOutN;
        curSol.workInN = workInN;
        curSol.workOutN = workOutN;
        curSol.countIn = countIn;
        curSol.countOut = countOut;
        curSol.cost = cost;

        if (curSol.cost < bestSol.cost ||
            (curSol.cost == bestSol.cost && curSol.countIn < bestSol.countIn) ||
            (curSol.cost == bestSol.cost && curSol.countIn == bestSol.countIn && curSol.countOut < bestSol.countOut)) {
            bestSol = curSol;
        }
    }

    if (bestSol.countOut == 0) {
        return std::make_tuple(0, 0, HwFullyConnectedTileInfo());
    }

    HwFullyConnectedTileInfo tiles;
    tiles.mode = bestSol.mode;
    tiles.numOutTiles = bestSol.countOut;
    tiles.numInSubTiles = bestSol.countIn;
    tiles.workInN = bestSol.workInN;
    tiles.workOutN = bestSol.workOutN;

    return std::make_tuple(std::max(bestSol.newInN, bestSol.countIn * bestSol.workInN), std::max(bestSol.newOutN, bestSol.countOut * bestSol.workOutN), tiles);
}

class HwFcRelayoutStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<HwFcRelayoutStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep,
            StageDataInfo<float>&) override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        orderInfo.setInput(inputEdge(0), input->desc().dimsOrder().createMovedDim(Dim::C, 2));
        orderInfo.setOutput(outputEdge(0), output->desc().dimsOrder().createMovedDim(Dim::C, 2));
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        auto output = outputEdge(0)->output();

        stridesInfo.setOutput(outputEdge(0), StridesRequirement().add(1, DimStride::Aligned));
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) override {
        batchInfo.setInput(inputEdge(0), BatchSupport::Split);
        batchInfo.setOutput(outputEdge(0), BatchSupport::Split);
    }

    StageSHAVEsRequirements getSHAVEsRequirementsImpl() const override {
        return StageSHAVEsRequirements::TwoOrOne;
    }

    void finalCheckImpl() const override {
        assertInputsOutputsTypes(this, {{DataType::FP16}}, {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeOldBuffer(this, serializer);
        output->serializeOldBuffer(this, serializer);
    }
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(hwFullyConnectedTiling);

    const auto& env = CompileEnv::get();

    for (const auto& origStage : model->getStages()) {
        if (origStage->type() != StageType::StubFullyConnected) {
            continue;
        }

        auto tryHW = origStage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        auto origInput = origStage->input(0);
        auto origWeights = origStage->input(1);
        auto origBiases = origStage->input(2);
        auto origOutput = origStage->output(0);

        auto withReLU = origStage->attrs().getOrDefault<bool>("withReLU", false);

        auto scaleFactor = origStage->attrs().getOrDefault<float>("scaleFactor", 1.0f);

        //
        // Repack input data if needed
        //

        auto hwInput = origInput;
        auto hwOutput = origOutput;

        Stage relayoutStage;

        if (hwInput->desc().numDims() > 2 &&
            (hwInput->desc().dim(Dim::W) != 1 || hwInput->desc().dim(Dim::H) != 1)) {
            auto newDesc = hwInput->desc();
            newDesc.setDim(Dim::W, 1);
            newDesc.setDim(Dim::H, 1);
            newDesc.setDim(Dim::C, hwInput->desc().totalDimSize());

            auto hwInputAsVec = model->duplicateData(
                hwInput,
                "@asVec",
                newDesc);

            relayoutStage = model->addNewStage<HwFcRelayoutStage>(
                origStage->name() + "@input-relayout",
                StageType::HwFcRelayout,
                origStage->origLayer(),
                {hwInput},
                {hwInputAsVec});

            hwInput = hwInputAsVec;
        }

        //
        // Try to find "best" tiling
        //

        //
        // Always use MODE_1_256
        //

        int extendedInputDimC = 0, extendedOutputDimC = 0;
        HwFullyConnectedTileInfo tiles;
        std::tie(extendedInputDimC, extendedOutputDimC, tiles) =
            splitFullyConnected(
                hwInput->desc().dim(Dim::C),
                hwOutput->desc().dim(Dim::C),
                {HwOpMode::MODE_1_256});

        //
        // Use SW stage if tiling optimization failed
        //

        if (tiles.numOutTiles == 0 ||
            calculateHwBufferSize(hwOutput->desc().dims()) > env.resources.cmxLimit) {
            origStage->attrs().set<bool>("tryHW", false);

            if (relayoutStage != nullptr) {
                model->removeStage(relayoutStage);
            }

            auto swOutput = origOutput;
            if (withReLU) {
                swOutput = model->addNewData(
                    origStage->name(),
                    origOutput->desc());
                swOutput->attrs().copyFrom(origOutput->attrs());

                model->replaceStageOutput(origStage->outputEdge(0), swOutput);

                _stageBuilder->addReLUStage(
                    model,
                    origStage->name() + "@ReLU",
                    origStage->origLayer(),
                    0.0,
                    swOutput,
                    origOutput);
            }

            continue;
        }

        model->disconnectStage(origStage);

        //
        // Expand input/output if needed
        //

        auto origInputDimC = hwInput->desc().dim(Dim::C);
        auto origOutputDimC = hwOutput->desc().dim(Dim::C);

        if (extendedInputDimC > origInputDimC) {
            auto newDesc = hwInput->desc();
            newDesc.setDim(Dim::C, extendedInputDimC);

            auto hwInputExtended = model->duplicateData(
                hwInput,
                "@extended",
                newDesc);

            _stageBuilder->addExpandStage(
                model,
                origStage->name() + "@expand-input",
                origStage->origLayer(),
                hwInput,
                hwInputExtended);

            hwInput = hwInputExtended;
        }

        auto hwWeights = origWeights->attrs().getOrDefault<Data>("hwWeights", nullptr);
        if (hwWeights == nullptr) {
            hwWeights = model->duplicateData(
                origWeights,
                "@HW",
                DataDesc({8, 1, extendedInputDimC, extendedOutputDimC / 8}),
                std::make_shared<HwWeightsContent>(
                    origWeights->content(),
                    DataDesc({1, 1, origInputDimC, origOutputDimC}),
                    hwInput->desc().dim(Dim::C)));

            if (scaleFactor != 1.0f) {
                auto hwWeightsScaled = model->duplicateData(
                    hwWeights,
                    formatString("@SCALE=%f", scaleFactor),
                    hwWeights->desc(),
                    scaleContent(hwWeights->content(), scaleFactor));
                hwWeightsScaled->attrs().getOrSet<float>("scaleFactor", 1.0f) *= scaleFactor;

                hwWeights = hwWeightsScaled;
            }

            origWeights->attrs().set<Data>("hwWeights", hwWeights);
        }

        auto hwBiases = origWeights->attrs().getOrDefault<Data>("hwBiases", nullptr);
        if (hwBiases == nullptr) {
            if (origBiases->usage() == DataUsage::Fake) {
                hwBiases = model->addFakeData();
            } else {
                auto origBiasesContent = origBiases->content();
                IE_ASSERT(origBiasesContent != nullptr);

                auto origBiasesPtr = origBiasesContent->get<fp16_t>();
                IE_ASSERT(origBiasesPtr != nullptr);

                auto hwBiasesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                    ie::Precision::FP16,
                    {static_cast<size_t>(extendedOutputDimC)},
                    ie::Layout::C));
                hwBiasesBlob->allocate();

                auto hwBiasesBlobPtr = hwBiasesBlob->buffer().as<fp16_t*>();
                IE_ASSERT(hwBiasesBlobPtr != nullptr);

                std::fill_n(hwBiasesBlobPtr, extendedOutputDimC, ie::PrecisionUtils::f32tof16(0.0f));
                std::copy_n(origBiasesPtr, origOutputDimC, hwBiasesBlobPtr);

                hwBiases = model->duplicateData(
                    origBiases,
                    "@HW",
                    DataDesc({extendedOutputDimC}),
                    ieBlobContent(hwBiasesBlob));

                if (scaleFactor != 1.0f) {
                    auto hwBiasesScaled = model->duplicateData(
                        hwBiases,
                        formatString("@SCALE=%f", scaleFactor),
                        hwBiases->desc(),
                        scaleContent(hwBiases->content(), scaleFactor));
                    hwBiasesScaled->attrs().getOrSet<float>("scaleFactor", 1.0f) *= scaleFactor;

                    hwBiases = hwBiasesScaled;
                }
            }

            origWeights->attrs().set<Data>("hwBiases", hwBiases);
        }

        Data hwScales = model->addFakeData();
        if (scaleFactor != 1.0f) {
            hwScales = origWeights->attrs().getOrDefault<Data>("hwScales", nullptr);

            if (hwScales == nullptr) {
                hwScales = model->addConstData(
                    origStage->name() + "@scales",
                    DataDesc({extendedOutputDimC}),
                    replicateContent(1.0f / scaleFactor, extendedOutputDimC));

                origWeights->attrs().set<Data>("hwScales", hwScales);
            }
        }

        auto hwStage = model->addNewStage<MyriadXHwStage>(
            origStage->name(),
            StageType::MyriadXHwOp,
            origStage->origLayer(),
            {hwInput, hwWeights, hwBiases, hwScales},
            {hwOutput});

        hwStage->attrs().set<HwOpType>("hwOpType", HwOpType::FC);

        hwStage->attrs().set("tiling", tiles);

        hwStage->attrs().set<bool>("withReLU", withReLU);

        hwStage->attrs().set<float>("scaleFactor", scaleFactor);

        //
        // Remove SW stage
        //

        model->removeStage(origStage);
    }
}

}  // namespace

Pass::Ptr PassManager::hwFullyConnectedTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
