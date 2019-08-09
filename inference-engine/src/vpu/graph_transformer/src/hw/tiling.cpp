// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/hw/tiling.hpp>

#include <tuple>
#include <string>
#include <algorithm>
#include <vector>
#include <limits>
#include <utility>

#include <vpu/hw/utility.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

//
// Tiling scheme
//

void printTo(std::ostream& os, const HwConvTileInfo& convTiles) {
    os << "[" << std::endl;
    os << "mode=" << convTiles.mode << std::endl;
    os << "numDescr=" << convTiles.numDescr << std::endl;
    os << "outChansPerDescr=" << convTiles.outChansPerDescr << std::endl;
    os << "lastOutChans=" << convTiles.lastOutChans << std::endl;
    os << "extendedInputDimC=" << convTiles.extendedInputDimC << std::endl;
    os << "extendedOutputDimC=" << convTiles.extendedOutputDimC << std::endl;
    os << "cost=" << convTiles.cost << std::endl;
    os << "]";
}

void printTo(DotLabel& lbl, const HwConvTileInfo& convTiles) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("mode", convTiles.mode);
    subLbl.appendPair("numDescr", convTiles.numDescr);
    subLbl.appendPair("outChansPerDescr", convTiles.outChansPerDescr);
    subLbl.appendPair("lastOutChans", convTiles.lastOutChans);
    subLbl.appendPair("extendedInputDimC", convTiles.extendedInputDimC);
    subLbl.appendPair("extendedOutputDimC", convTiles.extendedOutputDimC);
    subLbl.appendPair("cost", convTiles.cost);
}

void printTo(std::ostream& os, const HwPoolTileInfo& poolTiles) {
    os << "[" << std::endl;
    os << "mode=" << poolTiles.mode << std::endl;
    os << "numDescr=" << poolTiles.numDescr << std::endl;
    os << "chansPerDescr=" << poolTiles.chansPerDescr << std::endl;
    os << "]";
}

void printTo(DotLabel& lbl, const HwPoolTileInfo& poolTiles) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("mode", poolTiles.mode);
    subLbl.appendPair("numDescr", poolTiles.numDescr);
    subLbl.appendPair("chansPerDescr", poolTiles.chansPerDescr);
}

void printTo(std::ostream& os, const HwFullyConnectedTileInfo& fcTiles) {
    os << "[" << std::endl;
    os << "mode=" << fcTiles.mode << std::endl;
    os << "numOutTiles=" << fcTiles.numOutTiles << std::endl;
    os << "numInSubTiles=" << fcTiles.numInSubTiles << std::endl;
    os << "workInN=" << fcTiles.workInN << std::endl;
    os << "workOutN=" << fcTiles.workOutN << std::endl;
    os << "]";
}

void printTo(DotLabel& lbl, const HwFullyConnectedTileInfo& fcTiles) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("mode", fcTiles.mode);
    subLbl.appendPair("numOutTiles", fcTiles.numOutTiles);
    subLbl.appendPair("numInSubTiles", fcTiles.numInSubTiles);
    subLbl.appendPair("workInN", fcTiles.workInN);
    subLbl.appendPair("workOutN", fcTiles.workOutN);
}

//
// Input<->Output tile calculation
//

int calcOutputSize(
        int inputSize,
        int kernelSize, int kernelStride,
        int padBefore, int padAfter,
        bool useCeil) {
    if (useCeil) {
        return std::ceil(static_cast<double>(inputSize - kernelSize + padBefore + padAfter) / kernelStride + 1);
    } else {
        return (inputSize - kernelSize + padBefore + padAfter) / kernelStride + 1;
    }
}

//
// Plane tiles calculation.
//

SmallVector<HwPlaneTileInfo> splitIntoPlaneTilesWithPool(
        int inputSize,
        int kernelSize, int kernelStride,
        int pad,
        int maxOutputSize) {
    SmallVector<HwPlaneTileInfo> tiles;

    // This is very specific case for 3x3p1s1 convlution, followed by 2x2s2 pooling with even height
    IE_ASSERT(kernelSize == 3 && kernelStride == 1 && pad == 1);
    IE_ASSERT(inputSize % 2 == 0);

    // For this specific case, the outputSize is:
    int outputSize = inputSize / 2;

    IE_ASSERT(inputSize > 0);
    IE_ASSERT(outputSize > 0);

    if (outputSize > maxOutputSize) {
        if (maxOutputSize % 2 == 0) {
            --maxOutputSize;
        }
    }

    IE_ASSERT(maxOutputSize >= 2);

    int inputStartIndex = 0;
    int outputStartIndex = 0;

    while (true) {
        int inputEndIndex = std::min<int>(inputStartIndex + 2 * maxOutputSize, inputSize);
        int outputEndIndex = std::min<int>(outputStartIndex + maxOutputSize, outputSize);

        IE_ASSERT(inputEndIndex > inputStartIndex);
        IE_ASSERT(outputEndIndex > outputStartIndex);

        int trueInputNeeded = inputEndIndex - inputStartIndex;
        int outputWithJunk = outputEndIndex - outputStartIndex;
        int junkBefore = outputStartIndex > 0 ?  1 : 0;
        int junkAfter = outputEndIndex < outputSize ? 1 : 0;

        outputStartIndex += junkBefore;
        outputEndIndex -= junkAfter;

        HwPlaneTileInfo info;
        info.inputWithJunk = trueInputNeeded;
        info.outputWithJunk = outputWithJunk;
        info.outputJunkBefore = junkBefore;
        info.outputJunkAfter = junkAfter;
        info.inputStartIndex = inputStartIndex;
        info.inputEndIndex = inputEndIndex;
        info.outputStartIndex = outputStartIndex;
        info.outputEndIndex = outputEndIndex;

        tiles.emplace_back(info);

        if (outputEndIndex >= outputSize)
            break;

        auto newInputStartIndex = inputEndIndex - 4;
        auto newOutputStartIndex = outputEndIndex - 1;

        inputStartIndex = newInputStartIndex;
        outputStartIndex = newOutputStartIndex;
    }

    return tiles;
}

namespace {

// Note:
//
//   * [inputStartIndex, inputEndIndex): is the original range, without account for splits
//   * inputLinesBefore: specifies how many elements we need to remove from inputStartIndex to get the correct starting point
//   * junkOutputBefore: With starting point inputStartIndex - inputLinesBefore, this value contains the junk lines we need to discard
//   * [outputStartIndex, outputEndIndex): is the output range we are interested in generating, without the extra junk that might be there
//   * junkOutputBefore: is the junk contained before the outputStartIndex
//   * junkOutputAfter: is the junk contained after the outputEndIndex
//
// The output generated by the hardware is:
//
//   [outputStartIndex - junkOutputBefore, outputEndIndex + junkOutputAfter)
std::tuple<int, int, int, int, int, int, int, int>
    inputTileForOutputTile(
        int inputSize,
        int kernelSize, int kernelStride,
        int padBefore, int padAfter,
        int outputStartIndex, int outputEndIndex,
        bool alignInputTile) {
    // Negative value encodes the padding
    int inputStartIndex = outputStartIndex * kernelStride - padBefore;
    int inputEndIndex = (outputEndIndex - 1) * kernelStride + kernelSize - padBefore;

    int inputLinesBefore = 0;
    int junkOutputBefore = 0;

    if (inputStartIndex < 0) {
        // Negative inputStartIndex means that we use the original padding

        inputLinesBefore = 0;
        inputStartIndex = 0;
        if (outputStartIndex == 0) {
            junkOutputBefore = 0;
        } else {
            junkOutputBefore = outputStartIndex;
        }
    } else {
        // Non-negative inputStartIndex means that we either have no padding, or we are in the middle of the image

        // We reduce the inputStartIndex to the smallest non-negative integer
        inputLinesBefore = inputStartIndex;
        while (inputLinesBefore >= kernelStride) {
            inputLinesBefore -= kernelStride;
        }

        if (alignInputTile) {
            const int reqAlignment = 8;
            while ((inputLinesBefore < inputStartIndex) &&
                   (inputStartIndex - inputLinesBefore) % reqAlignment != 0) {
                ++inputLinesBefore;
            }
        }

        // Compute the junkOutputBefore
        junkOutputBefore = (inputLinesBefore + padBefore) / kernelStride;
    }

    int inputLinesAfter = 0;
    int junkOutputAfter = 0;

    if (inputEndIndex > inputSize) {
        // Larger inputEndIndex means that we use the original padding at the bottom of the image

        int paddingUsed = inputEndIndex - inputSize;

        inputLinesAfter = 0;
        inputEndIndex = inputSize;

        // The hardware will continue to compute output lines, until the kernel is just inside the padded image.
        junkOutputAfter = 0;
        while (paddingUsed + kernelStride <= padAfter) {
            paddingUsed += kernelStride;
            junkOutputAfter += 1;
        }
    } else {
        // This value of inputEndIndex means that we either have no padding, or we are in the middle of the image

        inputLinesAfter = 0;

        // Count how many kernels fit with the provided padding
        int paddingUsed = 0;
        junkOutputAfter = 0;
        while (paddingUsed + kernelStride <= padAfter) {
            paddingUsed += kernelStride;
            junkOutputAfter += 1;
        }
    }

    return std::make_tuple(inputStartIndex, inputEndIndex,
                           inputLinesBefore, inputLinesAfter,
                           outputStartIndex, outputEndIndex,
                           junkOutputBefore, junkOutputAfter);
}

int maximizeOutput(
        int inputSize, int maxOutputSize,
        int kernelSize, int kernelStride,
        int padBefore, int padAfter,
        int outputStartIndex, int outputEndIndex,
        bool alignInputTile,
        bool useCeil) {
    int outputSize = calcOutputSize(inputSize, kernelSize, kernelStride, padBefore, padAfter, useCeil);

    int _ = 0;
    int junkOutputBefore = 0, junkOutputAfter = 0;
    std::tie(_, _, _, _, _, _, junkOutputBefore, junkOutputAfter) =
        inputTileForOutputTile(inputSize, kernelSize, kernelStride, padBefore, padAfter, outputStartIndex, outputEndIndex, alignInputTile);

    int totalOutputSlice = junkOutputBefore + (outputEndIndex - outputStartIndex) + junkOutputAfter;

    auto isValid = [maxOutputSize, outputSize](int totalOutputSlice, int outputEndIndex) -> bool {
        return totalOutputSlice <= maxOutputSize && outputEndIndex <= outputSize;
    };

    int extraLines = 0;
    while (!isValid(totalOutputSlice, outputEndIndex + extraLines)) {
        extraLines -= 1;

        std::tie(_, _, _, _, _, _, junkOutputBefore, junkOutputAfter) =
            inputTileForOutputTile(inputSize, kernelSize, kernelStride, padBefore, padAfter, outputStartIndex, outputEndIndex + extraLines, alignInputTile);

        totalOutputSlice = junkOutputBefore + (outputEndIndex + extraLines - outputStartIndex) + junkOutputAfter;
    }

    return outputEndIndex + extraLines + !isValid(totalOutputSlice, outputEndIndex);
}

}  // namespace

SmallVector<HwPlaneTileInfo> splitIntoPlaneTiles(
        int inputSize, int outputSize,
        int kernelSize, int kernelStride,
        int padBefore, int padAfter,
        int maxOutputSize,
        bool alignInputTile,
        bool useCeil) {
    IE_ASSERT(inputSize > 0);
    IE_ASSERT(outputSize > 0);
    IE_ASSERT(maxOutputSize > 0);

    SmallVector<HwPlaneTileInfo> tiles;

    int outputStartIndex = 0;

    while (true) {
        int outputEndIndex = std::min<int>(outputSize, outputStartIndex + maxOutputSize);
        IE_ASSERT(outputEndIndex > outputStartIndex);

        int newOutputEndIndex = maximizeOutput(
            inputSize, maxOutputSize,
            kernelSize, kernelStride,
            padBefore, padAfter,
            outputStartIndex, outputEndIndex,
            alignInputTile,
            useCeil);
        if (newOutputEndIndex <= outputStartIndex) {
            return SmallVector<HwPlaneTileInfo>();
        }

        int inputStartIndex = 0, inputEndIndex = 0;
        int inputLinesBefore = 0, inputLinesAfter = 0;
        int junkOutputBefore = 0, junkOutputAfter = 0;
        std::tie(inputStartIndex, inputEndIndex,
                 inputLinesBefore, inputLinesAfter,
                 outputStartIndex, outputEndIndex,
                 junkOutputBefore, junkOutputAfter) =
            inputTileForOutputTile(inputSize, kernelSize, kernelStride, padBefore, padAfter, outputStartIndex, newOutputEndIndex, alignInputTile);

        IE_ASSERT(inputStartIndex >= 0);
        IE_ASSERT(inputEndIndex >= 0);
        IE_ASSERT(inputEndIndex > inputStartIndex);
        IE_ASSERT(inputLinesBefore >= 0);
        IE_ASSERT(inputLinesAfter >= 0);
        IE_ASSERT(inputStartIndex - inputLinesBefore >= 0);
        IE_ASSERT(inputEndIndex + inputLinesAfter >= 0);
        IE_ASSERT(inputEndIndex + inputLinesAfter <= inputSize);
        IE_ASSERT(inputLinesBefore + inputEndIndex - inputStartIndex + inputLinesAfter >= 0);
        IE_ASSERT(junkOutputBefore + outputEndIndex - outputStartIndex + junkOutputAfter >= 0);
        IE_ASSERT(junkOutputBefore >= 0);
        IE_ASSERT(junkOutputAfter >= 0);
        IE_ASSERT(outputStartIndex >= 0);
        IE_ASSERT(outputEndIndex >= 0);
        IE_ASSERT(outputEndIndex <= outputSize);

        HwPlaneTileInfo info;
        info.inputWithJunk = inputLinesBefore + inputEndIndex - inputStartIndex + inputLinesAfter;
        info.outputWithJunk = junkOutputBefore + outputEndIndex - outputStartIndex + junkOutputAfter;
        info.outputJunkBefore = junkOutputBefore;
        info.outputJunkAfter = junkOutputAfter;
        info.inputStartIndex = inputStartIndex - inputLinesBefore;
        info.inputEndIndex = inputEndIndex + inputLinesAfter;
        info.outputStartIndex = outputStartIndex;
        info.outputEndIndex = outputEndIndex;

        tiles.emplace_back(info);

        if (outputEndIndex >= outputSize)
            break;

        auto newOutputStartIndex = outputEndIndex;
        IE_ASSERT(newOutputStartIndex > outputStartIndex);

        outputStartIndex = newOutputStartIndex;
    }

    return tiles;
}

//
// HW Convolution tiling over output channels.
//

namespace {

// Returns (status, cost).
std::tuple<bool, int> checkHwConvMode(
        int inTileWidth, int inTileHeight, int inTileChannels,
        int outTileChannels,
        int kernelSizeX, int kernelSizeY,
        int kernelStride,
        HwOpMode mode) {
    if (inTileWidth > CNN_MAX_INPUT_WIDTH ||
        inTileHeight > CNN_MAX_INPUT_HEIGHT ||
        inTileChannels > CNN_MAX_INPUT_CHANNELS ||
        outTileChannels > CNN_MAX_OUTPUT_CHANNELS) {
        return std::make_tuple(false, 0);
    }

    auto noOfBlocks = 1 << static_cast<int>(mode);
    if (noOfBlocks > inTileChannels) {
        return std::make_tuple(false, 0);
    }

    auto inChansPerBlock = inTileChannels / noOfBlocks;
    if (inChansPerBlock > CNN_MAX_CHANNELS_PER_BLOCK) {
        return std::make_tuple(false, 0);
    }

    auto coeffPerWord = CNN_COEFF_PER_WORD_VALUES[static_cast<int32_t>(CNN_COEFF_TYPE)];
    auto coeffSetSize = kernelSizeX * kernelSizeY;
    auto coeffLPB = (inChansPerBlock * coeffSetSize + coeffPerWord - 1) / coeffPerWord;
    if (coeffLPB > CNN_MAX_COEFF_PER_BLOCK) {
        return std::make_tuple(false, 0);
    }

    auto bytesPerPixel = CNN_BYTES_PER_PIXEL[static_cast<int32_t>(CNN_DATA_TYPE)];
    auto pixelsPerCMXLine = 128 / (bytesPerPixel * 8);
    auto localLineStride = (inTileWidth + (pixelsPerCMXLine - 1)) / pixelsPerCMXLine;
    auto bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel;
    auto sizeOfBlock = CNN_MAX_BYTES >> static_cast<int>(mode);
    auto chanPerBlock = inTileChannels / noOfBlocks;
    if (chanPerBlock == 0) {
        return std::make_tuple(false, 0);
    }

    auto availableBytesPerChan = sizeOfBlock / chanPerBlock;
    auto linesPerChan = std::min(availableBytesPerChan / bytesPerLine, inTileHeight);
    auto minLines = std::min(kernelSizeY / 1 + (kernelStride + 1) + 1 + ((inTileWidth <= 8) ? 1 : 0), inTileHeight);
    if (minLines > linesPerChan) {
        return std::make_tuple(false, 0);
    }

    return std::make_tuple(true, (inTileChannels / noOfBlocks) * kernelSizeX * kernelSizeY + CNN_MODES_COST[static_cast<int32_t>(mode)]);
}

}  // namespace

HwConvTileInfo splitHwConvIntoOutChannelsTiles(
        int inTileWidth, int inTileHeight, int inTileChannels,
        int outTileChannels,
        int kernelSizeX, int kernelSizeY,
        int kernelStride) {
    struct Solution final {
        HwOpMode mode = HwOpMode::MODE_1_256;
        int extendedInputDimC = 0;
        int extendedOutputDimC = 0;
        int numDescr = 0;
        int outChansPerDescr = 0;
        int remOutChans = 0;
        int cost = std::numeric_limits<int>::max();
    };

    Solution bestSol;

    for (auto mode : CNN_MODES) {
        auto ramBlocks = 1 << static_cast<int>(mode);

        auto extendedInputDimC = alignVal(inTileChannels, ramBlocks);
        auto extendedOutputDimC = alignVal(outTileChannels, 8);

        auto outChansPerDescr = std::min(256 / ramBlocks, extendedOutputDimC);

        bool valid = false;
        int descCost = 0;
        std::tie(valid, descCost) = checkHwConvMode(
            inTileWidth, inTileHeight, extendedInputDimC,
            outChansPerDescr,
            kernelSizeX, kernelSizeY,
            kernelStride,
            mode);

        if (!valid) {
            continue;
        }

        auto numDescr = divUp(outTileChannels, outChansPerDescr);
        auto remOutChans = outTileChannels - (numDescr - 1) * outChansPerDescr;

        Solution curSol;
        curSol.mode = mode;
        curSol.extendedInputDimC = extendedInputDimC;
        curSol.extendedOutputDimC = extendedOutputDimC;
        curSol.numDescr = numDescr;
        curSol.outChansPerDescr = outChansPerDescr;
        curSol.remOutChans = remOutChans;
        curSol.cost = numDescr * descCost;

        if (curSol.cost < bestSol.cost || (curSol.cost == bestSol.cost && curSol.numDescr < bestSol.numDescr)) {
            bestSol = curSol;
        }
    }

    if (bestSol.numDescr == 0) {
        return HwConvTileInfo();
    }

    IE_ASSERT(bestSol.extendedInputDimC > 0);
    IE_ASSERT(bestSol.extendedOutputDimC > 0);
    IE_ASSERT(bestSol.numDescr > 0);
    IE_ASSERT(bestSol.outChansPerDescr > 0);

    HwConvTileInfo tileInfo;
    tileInfo.mode = bestSol.mode;
    tileInfo.numDescr = bestSol.numDescr;
    tileInfo.outChansPerDescr = bestSol.outChansPerDescr;
    tileInfo.lastOutChans = bestSol.remOutChans > 0 ? bestSol.remOutChans : bestSol.outChansPerDescr;
    tileInfo.extendedInputDimC = bestSol.extendedInputDimC;
    tileInfo.extendedOutputDimC = bestSol.extendedOutputDimC;
    tileInfo.cost = bestSol.cost;

    return tileInfo;
}

}  // namespace vpu
