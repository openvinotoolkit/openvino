// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include <vpu/model/data_contents/hw_weights_content.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <precision_utils.h>

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

namespace vpu {

namespace {

std::tuple<int, int, HwFullyConnectedTileInfo> splitFullyConnected(int inN, int outN,
    const SmallVector<HwOpMode>& modes = {
        HwOpMode::MODE_1_256,
        HwOpMode::MODE_2_128,
        HwOpMode::MODE_4_64,
        HwOpMode::MODE_8_32,
        HwOpMode::MODE_16_16}) {
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
    for (const auto& mode : modes) {
        const auto ramBlocks = static_cast<int>(1ULL << static_cast<std::size_t>(mode));
        const auto maxInN = ramBlocks * 256;
        const auto maxOutN = 256 / ramBlocks;
        const auto newInN = alignVal(inN, ramBlocks);
        const auto newOutN = alignVal(outN, 8);
        const auto workInN = std::min(newInN, maxInN);
        const auto workOutN = std::min(newOutN, maxOutN);

        if (workInN < ramBlocks) {
            continue;
        }

        const auto countIn = static_cast<int>(std::ceil(static_cast<double>(newInN) / workInN));
        const auto countOut = static_cast<int>(std::ceil(static_cast<double>(newOutN) / workOutN));
        const auto cost = countIn * countOut * (workInN / ramBlocks + CNN_MODES_COST[static_cast<int>(mode)]);

        Solution currSol;
        currSol.mode = mode;
        currSol.newInN = newInN;
        currSol.newOutN = newOutN;
        currSol.workInN = workInN;
        currSol.workOutN = workOutN;
        currSol.countIn = countIn;
        currSol.countOut = countOut;
        currSol.cost = cost;

        const auto currCost = std::make_tuple(currSol.cost, currSol.countIn, currSol.countOut);
        const auto bestCost = std::make_tuple(bestSol.cost, bestSol.countIn, bestSol.countOut);
        if (currCost < bestCost) {
            bestSol = currSol;
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

    return std::make_tuple(
        std::max(bestSol.newInN, bestSol.countIn * bestSol.workInN),
        std::max(bestSol.newOutN, bestSol.countOut * bestSol.workOutN),
        tiles);
}

class HwFcRelayoutStage final : public StageNode {
public:
    using StageNode::StageNode;

private:
    StagePtr cloneImpl() const override {
        return std::make_shared<HwFcRelayoutStage>(*this);
    }

    void propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) override {
        const auto& input = inputEdge(0)->input();
        const auto& output = outputEdge(0)->output();

        const auto& inputMovedDim = input->desc().dimsOrder().createMovedDim(Dim::C, 2);
        const auto& outputMovedDim = output->desc().dimsOrder().createMovedDim(Dim::C, 2);

        orderInfo.setInput(inputEdge(0), inputMovedDim);
        orderInfo.setOutput(outputEdge(0), outputMovedDim);
    }

    void getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) override {
        stridesInfo.setOutput(
            outputEdge(0),
            StridesRequirement().add(1, DimStride::Aligned));
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
        assertInputsOutputsTypes(this,
            {{DataType::FP16}},
            {{DataType::FP16}});
    }

    void serializeParamsImpl(BlobSerializer&) const override {
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input = inputEdge(0)->input();
        auto output = outputEdge(0)->output();

        input->serializeBuffer(serializer);
        output->serializeBuffer(serializer);
    }
};

Data createHWInput(const Model& model, const Stage& original, Stage& relayout) {
    auto hwInput = original->input(0);
    const auto& descriptor = hwInput->desc();
    if (descriptor.numDims() > 2 && (descriptor.dim(Dim::W) != 1 || descriptor.dim(Dim::H) != 1)) {
        auto newDesc = descriptor;
        newDesc.setDim(Dim::W, 1);
        newDesc.setDim(Dim::H, 1);
        newDesc.setDim(Dim::C, descriptor.totalDimSize());

        auto hwInputAsVec = model->duplicateData(hwInput, "@asVec", newDesc);

        relayout = model->addNewStage<HwFcRelayoutStage>(
            original->name() + "@input-relayout",
            StageType::HwFcRelayout,
            original->origLayer(),
            {hwInput},
            {hwInputAsVec});

        hwInput = hwInputAsVec;
    }

    return hwInput;
}

Data createHWWeights(const Model& model, const Stage& original, int hwInputDimC, int hwOutputDimC,
                     int extendedHWInputDimC, int extendedHWOutputDimC) {
    const auto& origWeights = original->input(1);

    auto hwWeights = origWeights->attrs().getOrDefault<Data>("hwWeights", nullptr);
    if (hwWeights == nullptr) {
        const auto& dataDescriptor = DataDesc{8, 1, extendedHWInputDimC, extendedHWOutputDimC / 8};
        const auto& contentDescriptor = DataDesc{1, 1, hwInputDimC, hwOutputDimC};

        const auto& content = std::make_shared<HwWeightsContent>(
            origWeights->content(),
            dataDescriptor,
            contentDescriptor,
            extendedHWInputDimC);

        hwWeights = model->duplicateData(origWeights, "@HW", dataDescriptor, content);

        // Store "hwWeights` attribute to share the same scales between different batches
        origWeights->attrs().set<Data>("hwWeights", hwWeights);
    }

    return hwWeights;
}

Data createHWBiases(const Model& model, const Stage& original, int hwOutputDimC, int extendedHWOutputDimC) {
    const auto& origBiases = original->input(2);

    auto hwBiases = origBiases->attrs().getOrDefault<Data>("hwBiases", nullptr);
    if (hwBiases == nullptr) {
        if (origBiases->usage() == DataUsage::Fake) {
            hwBiases = model->addFakeData();
        } else {
            const auto& origBiasesContent = origBiases->content();
            IE_ASSERT(origBiasesContent != nullptr);

            const auto& origBiasesPtr = origBiasesContent->get<fp16_t>();
            IE_ASSERT(origBiasesPtr != nullptr);

            auto hwBiasesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                ie::Precision::FP16,
                {static_cast<size_t>(extendedHWOutputDimC)},
                ie::Layout::C));
            hwBiasesBlob->allocate();

            auto hwBiasesBlobPtr = hwBiasesBlob->buffer().as<fp16_t*>();
            IE_ASSERT(hwBiasesBlobPtr != nullptr);

            std::fill_n(hwBiasesBlobPtr, extendedHWOutputDimC, ie::PrecisionUtils::f32tof16(0.0f));
            std::copy_n(origBiasesPtr, hwOutputDimC, hwBiasesBlobPtr);

            hwBiases = model->duplicateData(
                origBiases,
                "@HW",
                DataDesc({extendedHWOutputDimC}),
                ieBlobContent(hwBiasesBlob));
        }

        // Store "hwBiases` attribute to share the same scales between different batches
        origBiases->attrs().set<Data>("hwBiases", hwBiases);
    }

    return hwBiases;
}

Data createHWScales(const Model& model, const Stage& original, int hwOutputDimC, int extendedHWOutputDimC) {
    const auto& origScales = original->input(3);

    auto hwScales = origScales->attrs().getOrDefault<Data>("hwScales", nullptr);
    if (hwScales == nullptr) {
        if (origScales->usage() == DataUsage::Fake) {
            hwScales = model->addFakeData();
        } else {
            const auto& origScalesContent = origScales->content();
            IE_ASSERT(origScalesContent != nullptr);

            const auto& origScalesPtr = origScalesContent->get<fp16_t>();
            IE_ASSERT(origScalesPtr != nullptr);

            auto hwScalesBlob = ie::make_shared_blob<fp16_t>(InferenceEngine::TensorDesc(
                ie::Precision::FP16,
                {static_cast<size_t>(extendedHWOutputDimC)},
                ie::Layout::C));
            hwScalesBlob->allocate();

            auto hwScalesBlobPtr = hwScalesBlob->buffer().as<fp16_t*>();
            IE_ASSERT(hwScalesBlobPtr != nullptr);

            std::fill_n(hwScalesBlobPtr, extendedHWOutputDimC, ie::PrecisionUtils::f32tof16(0.0f));
            std::copy_n(origScalesPtr, hwOutputDimC, hwScalesBlobPtr);

            hwScales = model->duplicateData(
                origScales,
                "@HW",
                DataDesc({extendedHWOutputDimC}),
                ieBlobContent(hwScalesBlob));
        }

        // Store "hwScales` attribute to share the same scales between different batches
        origScales->attrs().set<Data>("hwScales", hwScales);
    }

    return hwScales;
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override {
        VPU_PROFILE(hwFullyConnectedTiling);

        const auto& env = CompileEnv::get();
        const auto cmxLimit = tilingCMXLimit(env.resources.numCMXSlices);

        for (const auto& origStage : model->getStages()) {
            if (origStage->type() != StageType::StubFullyConnected) {
                continue;
            }

            if (!origStage->attrs().getOrDefault<bool>("tryHW", false)) {
                continue;
            }

            const auto& hwOutput = origStage->output(0);
            const auto& hwOutputDesc = hwOutput->desc();

            Stage relayoutStage;
            auto hwInput = createHWInput(model, origStage, relayoutStage);

            const auto& hwInputDimC = hwInput->desc().dim(Dim::C);
            const auto& hwOutputDimC = hwOutputDesc.dim(Dim::C);
            const auto& split = splitFullyConnected(hwInputDimC, hwOutputDimC, {HwOpMode::MODE_1_256});

            const auto& withReLU = origStage->attrs().getOrDefault<bool>("withReLU", false);
            const auto& tiles = std::get<2>(split);
            if (tiles.numOutTiles == 0 || calculateHwBufferSize(hwOutputDesc.dims()) > cmxLimit) {
                fallbackToSW(model, origStage, relayoutStage, withReLU);
                continue;
            }

            model->disconnectStage(origStage);

            const auto& extendedInputDimC = std::get<0>(split);
            if (extendedInputDimC > hwInputDimC) {
                hwInput = extendedHWInput(model, origStage, hwInput, extendedInputDimC);
            }

            const auto& extendedOutputDimC = std::get<1>(split);
            const auto& hwWeights = createHWWeights(
                model,
                origStage,
                hwInputDimC,
                hwOutputDimC,
                extendedInputDimC,
                extendedOutputDimC);

            const auto& hwBiases = createHWBiases(model, origStage, hwOutputDimC, extendedOutputDimC);
            const auto& hwScales = createHWScales(model, origStage, hwOutputDimC, extendedOutputDimC);

            auto hwStage = model->addNewStage<MyriadXHwStage>(
                origStage->name(),
                StageType::MyriadXHwOp,
                origStage->origLayer(),
                {hwInput, hwWeights, hwBiases, hwScales},
                {origStage->output(0)});

            hwStage->attrs().set<HwOpType>("hwOpType", HwOpType::FC);
            hwStage->attrs().set("tiling", tiles);
            hwStage->attrs().set<bool>("withReLU", withReLU);
            hwStage->attrs().set<float>("scaleFactor", origStage->attrs().getOrDefault<float>("scaleFactor", 1.0f));

            model->removeStage(origStage);
        }
    }

private:
    Data extendedHWInput(const Model& model, const Stage& original, const Data& hwInput, int extendedChannels) {
        auto newDesc = hwInput->desc();
        newDesc.setDim(Dim::C, extendedChannels);

        auto hwInputExtended = model->duplicateData(hwInput, "@extended", newDesc);

        _stageBuilder->addExpandStage(
            model,
            original->name() + "@expand-input",
            original->origLayer(),
            hwInput,
            hwInputExtended);

        return hwInputExtended;
    }

    void fallbackToSW(const Model& model, const Stage& original, const Stage& relayout, bool withReLU) const {
        const auto& output = original->output(0);
        original->attrs().set<bool>("tryHW", false);

        if (relayout != nullptr) {
            model->removeStage(relayout);
        }

        if (withReLU) {
            auto swOutput = model->addNewData(original->name(), output->desc());
            swOutput->attrs().copyFrom(output->attrs());

            model->replaceStageOutput(original->outputEdge(0), swOutput);

            _stageBuilder->addReLUStage(
                model,
                original->name() + "@ReLU",
                original->origLayer(),
                // LeakyReLU cannot be merged into FullyConnected
                0.0,
                swOutput,
                output);
        }
    }

    StageBuilder::Ptr _stageBuilder;
};

}  // namespace

Pass::Ptr PassManager::hwFullyConnectedTiling() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
