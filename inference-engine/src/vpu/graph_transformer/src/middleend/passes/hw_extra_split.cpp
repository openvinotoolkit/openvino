// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/tiling.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include <vpu/middleend/hw/conv_tiling/hw_convolution_tiler.hpp>
#include <vpu/middleend/hw/conv_tiling/hw_stage_tiler.hpp>
#include <vpu/model/data_contents/hw_const_data_content.hpp>

#include <precision_utils.h>

#include <utility>
#include <memory>
#include <set>
#include <string>
#include <algorithm>
#include <tuple>
#include <map>
#include <vector>
#include <limits>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override;

private:
    void splitStagesWithOneDescriptor(const Model& model);
    void splitStagesWithManyDescriptors(const Model& model);

    Data splitWeights(
        const Model& model,
        const Data& weights,
        const std::string& postfix,
        const int startChannel,
        const int numChannels);

    Data splitBiases(
        const Model& model,
        const Data& biases,
        const std::string& postfix,
        const int startChannel,
        const int numChannels);

    Data splitScales(
        const Model& model,
        const Data& scales,
        const std::string& postfix,
        const int startChannel,
        const int numChannels);

    Data reuseData(
        const Data& data,
        const int startChannel,
        const int numChannels);

    void splitHwConv(
        const Stage& stage,
        const Model& model,
        const std::string& postfixDescription,
        const std::vector<ConvTileSlice>& tileInfos);

    Data splitHwPool(
        const Stage& stage,
        const Model& model,
        const std::string& postfixDescription,
        const DataVector& inputs,
        const Data& output,
        const int currentDim,
        const int numSplits,
        const int channelsPerSplit,
        const int lastSplitChannels,
        const std::pair<HwPoolTileInfo, HwPoolTileInfo>& tiles);

    std::string getPostfix(
        const std::string description,
        const int index,
        const int total);

    std::vector<HwConvTileInfo> splitHwConvInMultipleOutChannelsTiles(
        const uint32_t inTileWidth, const uint32_t inTileHeight, const uint32_t inTileChannels,
        const uint32_t outTileChannels, const uint32_t tileDivision,
        const uint32_t kernelSizeX, const uint32_t kernelSizeY,
        const uint32_t kernelStride);

private:
    StageBuilder::Ptr _stageBuilder;

    struct LexicographicalCompareByData {
        bool operator() (const Data& data1, const Data& data2) const {
            auto infoData1 = std::make_tuple(data1->name(), data1->desc().dims());
            auto infoData2 = std::make_tuple(data2->name(), data2->desc().dims());
            if (infoData1 != infoData2)
                return infoData1 < infoData2;

            const auto size = data1->content()->byteSize() / sizeof(fp16_t);

            const auto content1 = data1->content()->get<fp16_t>();
            const auto content2 = data2->content()->get<fp16_t>();

            return std::lexicographical_compare(content1, content1 + size, content2, content2 + size);
        }
    };
    std::map<Data, DataSlices, LexicographicalCompareByData> _splitConstData;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(hwExtraSplit);

    splitStagesWithOneDescriptor(model);
    splitStagesWithManyDescriptors(model);
}

void PassImpl::splitStagesWithOneDescriptor(
    const Model& model) {
    // TODO: value chosen arbitrarily; to be investigated
    const int numSplits = 2;

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW)
            continue;

        const auto opType = stage->attrs().get<HwOpType>("hwOpType");

        if (opType == HwOpType::CONV || opType == HwOpType::CONV_POOL) {
            const auto tiling = stage->attrs().get<HwConvTileInfo>("tiling");

            if (tiling.numDescr > 1)
                continue;

            if (tiling.lastOutChans <= 8)
                continue;

            const auto input = stage->input(0);
            const auto inputDims = input->desc().dims();

            const auto output = stage->output(0);
            const auto outputDims = output->desc().dims();

            const auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
            const auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
            const auto kernelStride = stage->attrs().get<int>("kernelStride");

            const auto newTiles = splitHwConvInMultipleOutChannelsTiles(
                            inputDims[Dim::W], inputDims[Dim::H], inputDims[Dim::C],
                            outputDims[Dim::C], numSplits,
                            kernelSizeX,
                            kernelSizeY,
                            kernelStride);

            if (newTiles.size() < numSplits)
                continue;

            std::vector<ConvTileSlice> tileInfos;
            for (int i = 0; i < newTiles.size(); ++i) {
                const auto tile = newTiles[i];

                const auto startChannel = i * tile.outChansPerDescr;
                const auto numChannels = tile.lastOutChans;
                const auto slice = Slice(startChannel, numChannels);

                tileInfos.emplace_back(tile, slice);
            }

            splitHwConv(stage, model, "@extra_sok", tileInfos);
        } else if (opType == HwOpType::POOL) {
            const auto tiling = stage->attrs().get<HwPoolTileInfo>("tiling");

            if (tiling.numDescr > 1)
                continue;

            const auto inputs = stage->inputs() | asSmallVector();
            const auto output = stage->output(0);
            const auto channels = output->desc().dim(Dim::C);

            if (channels <= 1)
                continue;

            model->disconnectStage(stage);

            DataVector dimOutputs;
            const auto numDims = output->desc().dim(Dim::N, 1);

            for (int dim = 0; dim < numDims; ++dim) {
                const auto channelsPerSplit = divUp(channels, numSplits);
                const auto lastSplitChannels = channels - channelsPerSplit * (numSplits-1);

                std::pair<HwPoolTileInfo, HwPoolTileInfo> tiles = std::make_pair(tiling, tiling);
                tiles.first.numDescr  = 1;
                tiles.second.numDescr = 1;
                tiles.first.chansPerDescr  = channelsPerSplit;
                tiles.second.chansPerDescr = lastSplitChannels;

                const auto dimOutput = splitHwPool(stage, model, "@extra_sok", inputs, output, dim, numSplits, channelsPerSplit, lastSplitChannels, tiles);
                dimOutputs.push_back(dimOutput);
            }

            _stageBuilder->addConcatStage(
                model,
                stage->name() + "@concat-output",
                stage->origLayer(),
                Dim::N,
                dimOutputs,
                output);

            model->removeStage(stage);
        }
    }
}

void PassImpl::splitStagesWithManyDescriptors(
        const Model& model) {
    // TODO: value chosen arbitrarily; to be investigated
    const int descLimit = 8;

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::HW)
            continue;

        const auto opType = stage->attrs().get<HwOpType>("hwOpType");

        if (opType == HwOpType::CONV || opType == HwOpType::CONV_POOL) {
            const auto tiling = stage->attrs().get<HwConvTileInfo>("tiling");

            if (tiling.numDescr <= descLimit)
                continue;

            const auto numSplits = (tiling.numDescr + (descLimit - 1)) / descLimit;
            const auto remDesc = tiling.numDescr % descLimit;
            const auto lastSplitNumDesc = (remDesc == 0) ? descLimit : remDesc;

            std::vector<ConvTileSlice> tileInfos;
            for (int i = 0; i < numSplits; ++i) {
                HwConvTileInfo tile = tiling;
                tile.numDescr = (i == numSplits-1) ? lastSplitNumDesc : descLimit;
                if (i < numSplits-1)
                    tile.lastOutChans = tile.outChansPerDescr;

                const auto startChannel = i * tiling.outChansPerDescr * descLimit;
                const auto numChannels = (i == numSplits-1) ?
                                        tiling.outChansPerDescr * (lastSplitNumDesc - 1) + tiling.lastOutChans :
                                        tiling.outChansPerDescr * descLimit;
                const auto slice = Slice(startChannel, numChannels);

                tileInfos.emplace_back(tile, slice);
            }

            splitHwConv(stage, model, "@extra_split", tileInfos);
        } else if (opType == HwOpType::POOL) {
            const auto tiling = stage->attrs().get<HwPoolTileInfo>("tiling");

            if (tiling.numDescr <= descLimit)
                continue;

            const auto inputs = stage->inputs() | asSmallVector();
            const auto output = stage->output(0);

            model->disconnectStage(stage);

            DataVector dimOutputs;

            const auto numDims = output->desc().dim(Dim::N, 1);
            const auto descPerDim = tiling.numDescr / numDims;

            for (int dim = 0; dim < numDims; ++dim) {
                const auto numSplits = divUp(descPerDim, descLimit);
                const auto remDesc = descPerDim % descLimit;
                const auto lastSplitNumDesc = (remDesc == 0) ? descLimit : remDesc;

                const auto totalChannels = output->desc().dim(Dim::C);
                const auto lastDescChans = totalChannels - (descPerDim-1) * tiling.chansPerDescr;

                const auto channelsPerSplit  = tiling.chansPerDescr * descLimit;
                const auto lastSplitChannels = (lastSplitNumDesc-1) * tiling.chansPerDescr + lastDescChans;

                std::pair<HwPoolTileInfo, HwPoolTileInfo> tiles = std::make_pair(tiling, tiling);
                tiles.first.numDescr  = descLimit;
                tiles.second.numDescr = lastSplitNumDesc;

                const auto dimOutput = splitHwPool(stage, model, "@extra_split", inputs, output, dim, numSplits, channelsPerSplit, lastSplitChannels, tiles);
                dimOutputs.push_back(dimOutput);
            }

            _stageBuilder->addConcatStage(
                model,
                stage->name() + "@concat-output",
                stage->origLayer(),
                Dim::N,
                dimOutputs,
                output);

            model->removeStage(stage);
        }
    }
}

Data PassImpl::splitWeights(
    const Model& model,
    const Data& weights,
    const std::string& postfix,
    const int startChannel,
    const int numChannels) {

    const auto preSplitWeights = reuseData(weights, startChannel, numChannels);
    if (preSplitWeights != weights)
        return preSplitWeights;

    auto weightsDesc = weights->desc();
    const auto vectorSize = weightsDesc.dims()[Dim::W];

    const std::map<Dim, Slice> dimSlices = {{Dim::N, Slice(startChannel, numChannels)}};
    const auto content = std::make_shared<HwConstData>(
        weights->content(),
        weights->desc(),
        weightsDesc,
        dimSlices);

    weightsDesc.setDim(Dim::N, alignVal(numChannels, 8) / vectorSize);
    const auto newWeights = model->duplicateData(weights, postfix, weightsDesc, content);

    _splitConstData[weights].emplace_back(newWeights, Slice(startChannel, numChannels));

    return newWeights;
}

Data PassImpl::splitBiases(
    const Model& model,
    const Data& biases,
    const std::string& postfix,
    const int startChannel,
    const int numChannels) {

    if (!biases->content()) return biases;

    const auto preSplitBiases = reuseData(biases, startChannel, numChannels);
    if (preSplitBiases != biases)
        return preSplitBiases;

    auto newBiasesDesc = biases->desc();
    newBiasesDesc.setDim(Dim::C, numChannels);

    const std::map<Dim, Slice> dimSlices = {{Dim::C, Slice(startChannel, numChannels)}};
    const auto biasesContent = std::make_shared<HwConstData>(
        biases->content(),
        biases->desc(),
        newBiasesDesc,
        dimSlices);
    const auto newBiases = model->duplicateData(biases, postfix, newBiasesDesc, biasesContent);

    _splitConstData[biases].emplace_back(newBiases, Slice(startChannel, numChannels));

    return newBiases;
}

Data PassImpl::splitScales(
    const Model& model,
    const Data& scales,
    const std::string& postfix,
    const int startChannel,
    const int numChannels) {

    if (!scales->content()) return scales;

    const auto preSplitScales = reuseData(scales, startChannel, numChannels);
    if (preSplitScales != scales)
        return preSplitScales;

    auto newScalesDesc = scales->desc();
    newScalesDesc.setDim(Dim::C, numChannels);

    const std::map<Dim, Slice> dimSlices = {{Dim::C, Slice(startChannel, numChannels)}};
    const auto scalesContent = std::make_shared<HwConstData>(
        scales->content(),
        scales->desc(),
        newScalesDesc,
        dimSlices);
    const auto newScales = model->duplicateData(scales, postfix, newScalesDesc, scalesContent);

    _splitConstData[scales].emplace_back(newScales, Slice(startChannel, numChannels));

    return newScales;
}

Data PassImpl::reuseData(
    const Data& data,
    const int startChannel,
    const int numChannels) {

    const auto splitData = _splitConstData.find(data);
    if (splitData != _splitConstData.end()) {
        const auto dataSlices = splitData->second;

        const auto slice = std::find_if(
                dataSlices.begin(), dataSlices.end(), [&](DataSlice dataSlice) {
                    return dataSlice.slice.start == startChannel &&
                        dataSlice.slice.size == numChannels;
                });

        if (slice != dataSlices.end())
            return slice->data;
    }

    return data;
}

void PassImpl::splitHwConv(
    const Stage& stage,
    const Model& model,
    const std::string& postfixDescription,
    const std::vector<ConvTileSlice>& tileInfos) {
    const auto input   = stage->input(0);
    const auto weights = stage->input(1);
    const auto biases  = stage->input(2);
    const auto scales  = stage->input(3);
    const auto output  = stage->output(0);

    model->disconnectStage(stage);

    DataVector newOutputs;

    for (int i = 0; i < tileInfos.size(); ++i) {
        const auto tileInfo = tileInfos[i];
        const auto tile = tileInfo.tile;

        const std::string postfix = getPostfix(postfixDescription, i+1, static_cast<int>(tileInfos.size()));

        const auto startChannel = tileInfo.slice.start;
        const auto numChannels = static_cast<int>(tileInfo.slice.size);

        const auto newWeights = splitWeights(model, weights, postfix, startChannel, numChannels);
        const auto newBiases = splitBiases(model, biases, postfix, startChannel, numChannels);
        const auto newScales = splitScales(model, scales, postfix, startChannel, numChannels);

        auto newOutputDesc = output->desc();
        newOutputDesc.setDim(Dim::C, numChannels);
        const auto newOutput = model->duplicateData(output, postfix, newOutputDesc);

        auto newStage = model->duplicateStage(
            stage,
            postfix,
            {input, newWeights, newBiases, newScales},
            {newOutput});
        newOutputs.emplace_back(newOutput);

        newStage->attrs().set<HwConvTileInfo>("tiling", tile);
    }

    _stageBuilder->addConcatStage(
        model,
        stage->name() + "@concat-output",
        stage->origLayer(),
        Dim::C,
        newOutputs,
        output);

    model->removeStage(stage);
}

Data PassImpl::splitHwPool(
        const Stage& stage,
        const Model& model,
        const std::string& postfixDescription,
        const DataVector& inputs,
        const Data& output,
        const int currentDim,
        const int numSplits,
        const int channelsPerSplit,
        const int lastSplitChannels,
        const std::pair<HwPoolTileInfo, HwPoolTileInfo>& tiles) {
    const auto input   = inputs[0];
    const auto weights = inputs[1];
    const auto biases  = inputs[2];
    const auto scales  = inputs[3];

    std::vector<DimValues> inputOffsets;
    DataVector newInputs;
    DataVector newOutputs;

    int currentChannels = 0;

    for (int i = 0; i < numSplits; ++i) {
        const std::string postfix = getPostfix(postfixDescription, i+1, numSplits);

        const auto splitChannels = (i == numSplits-1) ? lastSplitChannels : channelsPerSplit;

        auto newInputDesc  = input->desc();
        newInputDesc.setDim(Dim::N, 1);
        newInputDesc.setDim(Dim::C, splitChannels);
        const auto newInput = model->duplicateData(output, postfix, newInputDesc);

        auto newOutputDesc = output->desc();
        newOutputDesc.setDim(Dim::N, 1);
        newOutputDesc.setDim(Dim::C, splitChannels);
        const auto newOutput = model->duplicateData(output, postfix, newOutputDesc);

        auto newStage = model->duplicateStage(
            stage,
            postfix,
            {newInput, weights, biases, scales},
            {newOutput});

        newInputs.emplace_back(newInput);
        newOutputs.emplace_back(newOutput);

        inputOffsets.emplace_back(DimValues({
            {Dim::N, currentDim},
            {Dim::C, currentChannels}}));
        currentChannels += splitChannels;

        HwPoolTileInfo tile = (i == numSplits-1) ? tiles.second : tiles.first;
        newStage->attrs().set<HwPoolTileInfo>("tiling", tile);
    }

    _stageBuilder->addSplitStage(
        model,
        stage->name() + "@dim-split",
        stage->origLayer(),
        std::move(inputOffsets),
        input,
        newInputs);

    auto newOutputDesc = output->desc();
    newOutputDesc.setDim(Dim::N, 1);
    const auto newOutput = model->duplicateData(output, "@dim-concat", newOutputDesc);

    _stageBuilder->addConcatStage(
        model,
        stage->name() + "@dim-concat",
        stage->origLayer(),
        Dim::C,
        newOutputs,
        newOutput);

    return newOutput;
}

std::string PassImpl::getPostfix(
    const std::string description,
    const int index,
    const int total) {
    std::ostringstream ostr;
    ostr << description << "="
            << std::setw(2) << std::setfill('0') << index
            << "/"
            << std::setw(2) << std::setfill('0') << total;
    return ostr.str();
}

std::vector<HwConvTileInfo> PassImpl::splitHwConvInMultipleOutChannelsTiles(
        const uint32_t inTileWidth, const uint32_t inTileHeight, const uint32_t inTileChannels,
        const uint32_t outTileChannels, const uint32_t tileDivision,
        const uint32_t kernelSizeX, const uint32_t kernelSizeY,
        const uint32_t kernelStride) {
    std::vector<HwConvTileInfo> bestSol;

    for (auto mode : CNN_MODES) {
        const auto inChansPerBlock  = 1U << static_cast<int>(mode);

        const auto extendedInputDimC = alignVal(inTileChannels, inChansPerBlock);
        const auto extendedOutputDimC = alignVal(outTileChannels, 8U);

        const auto outChansPerDescr = alignVal(outTileChannels / tileDivision, 8U);

        const auto numDescr = divUp(outTileChannels, outChansPerDescr);
        const auto remOutChans = outTileChannels - (numDescr - 1) * outChansPerDescr;

        const auto maxOutChans = 256U >> static_cast<int>(mode);

        if (extendedInputDimC == 0 || extendedOutputDimC == 0 || outChansPerDescr == 0 || numDescr == 0)
            continue;

        if (outChansPerDescr > maxOutChans)
            continue;

        if (inTileChannels % inChansPerBlock != 0)
            continue;

        const bool valid = checkConvHWRestrictions(
            inTileWidth, inTileHeight, inTileChannels, outChansPerDescr,
            kernelSizeX, kernelSizeY, kernelStride, mode);
        if (!valid) {
            continue;
        }

        const int descCost = (extendedInputDimC / inChansPerBlock) * kernelSizeX * kernelSizeY +
            CNN_MODES_COST[static_cast<int32_t>(mode)];

        const auto bestCost = (bestSol.size() > 0) ? bestSol.front().cost : std::numeric_limits<int>::max();

        if (descCost < bestCost || (descCost == bestCost && numDescr < bestSol.size())) {
            HwConvTileInfo tileInfo;
            tileInfo.mode = mode;
            tileInfo.extendedInputDimC = extendedInputDimC;
            tileInfo.extendedOutputDimC = extendedOutputDimC;
            tileInfo.numDescr = 1;
            tileInfo.outChansPerDescr = outChansPerDescr;
            tileInfo.lastOutChans = outChansPerDescr;
            tileInfo.cost = descCost;

            bestSol.clear();
            for (uint32_t i = 0; i < numDescr; ++i) {
                if (i == numDescr-1)
                    tileInfo.lastOutChans = (remOutChans > 0) ? remOutChans : outChansPerDescr;

                bestSol.push_back(tileInfo);
            }
        }
    }

    return bestSol;
}

}  // namespace

Pass::Ptr PassManager::hwExtraSplit() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
