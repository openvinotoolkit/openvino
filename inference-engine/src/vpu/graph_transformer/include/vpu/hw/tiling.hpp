// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include <array>
#include <limits>

#include <vpu/backend/blob_format.hpp>
#include <vpu/model/data.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>

namespace vpu {

//
// Common constants
//

const HwDataMode CNN_DATA_TYPE = HwDataMode::FP16;
const HwCoeffMode CNN_COEFF_TYPE = HwCoeffMode::FP16;

const std::array<int, 2> CNN_COEFF_PER_WORD_VALUES{1, 2};
const std::array<int, 2> CNN_BYTES_PER_PIXEL{2, 1};

const std::array<HwOpMode, 5> CNN_MODES{HwOpMode::MODE_1_256, HwOpMode::MODE_2_128, HwOpMode::MODE_4_64, HwOpMode::MODE_8_32, HwOpMode::MODE_16_16};
const std::array<int, 5> CNN_MODES_COST{0, 5, 11, 19, 31};

const int CNN_MAX_INPUT_WIDTH = 4096;
const int CNN_MAX_INPUT_HEIGHT = 4096;
const int CNN_MAX_INPUT_CHANNELS = 2048;
const int CNN_MAX_OUTPUT_CHANNELS = 2048;

const int CNN_MAX_BYTES = 128 * 1024;
const int CNN_MAX_CHANNELS_PER_BLOCK = 2048;
const int CNN_MAX_COEFF_PER_BLOCK = 256;

//
// Tiling scheme
//

struct HwPlaneTileInfo final {
    int inputWithJunk = 0, outputWithJunk = 0;
    int outputJunkBefore = 0, outputJunkAfter = 0;
    int inputStartIndex = 0, inputEndIndex = 0;
    int outputStartIndex = 0, outputEndIndex = 0;
};

template <class Tiles> struct HwChannelTile;
template <class Tiles> using HwChannelTilePtr = std::shared_ptr<HwChannelTile<Tiles>>;

template <class Tiles> struct HwPlaneTile;
template <class Tiles> using HwPlaneTilePtr = std::shared_ptr<HwPlaneTile<Tiles>>;
template <class Tiles> using HwPlaneTileWeakPtr = std::weak_ptr<HwPlaneTile<Tiles>>;

template <class Tiles> struct HwTiling;
template <class Tiles> using HwTilingPtr = std::shared_ptr<HwTiling<Tiles>>;
template <class Tiles> using HwTilingWeakPtr = std::weak_ptr<HwTiling<Tiles>>;

template <class Tiles>
struct HwChannelTile final {
    HwPlaneTileWeakPtr<Tiles> parent;

    int socInd = 0;

    int channelStartIndex = 0;
    int numInputChannels = 0;

    int extendedInputDimC = 0;
    int extendedOutputDimC = 0;

    Tiles finalTiles;
};

template <class Tiles>
struct HwPlaneTile final {
    HwTilingWeakPtr<Tiles> parent;

    int sohInd = 0;
    int sowInd = 0;

    HwPlaneTileInfo heightInfo = {};
    HwPlaneTileInfo widthInfo = {};

    SmallVector<HwChannelTilePtr<Tiles>> channelTiles;
};

template <class Tiles>
struct HwTiling final {
    int sohTiles = 0;
    int sowTiles = 0;
    int socTiles = 0;

    SmallVector<HwPlaneTilePtr<Tiles>> planeTiles;
};

template <class Tiles>
void printTo(std::ostream& os, const HwTilingPtr<Tiles>& tiling) {
    os << "[" << std::endl;
    os << "sohTiles=" << tiling->sohTiles << std::endl;
    os << "sowTiles=" << tiling->sowTiles << std::endl;
    os << "socTiles=" << tiling->socTiles << std::endl;
    os << "]";
}

template <class Tiles>
void printTo(DotLabel& lbl, const HwTilingPtr<Tiles>& tiling) {
    DotLabel subLbl(lbl);
    subLbl.appendPair("sohTiles", tiling->sohTiles);
    subLbl.appendPair("sowTiles", tiling->sowTiles);
    subLbl.appendPair("socTiles", tiling->socTiles);
}

template <class Tiles>
std::string getChannelTilePostfix(const HwChannelTilePtr<Tiles>& channelTile) {
    auto planeTile = channelTile->parent.lock();
    IE_ASSERT(planeTile != nullptr);

    auto tiling = planeTile->parent.lock();
    IE_ASSERT(tiling != nullptr);

    std::ostringstream ostr;

    if (tiling->socTiles > 1)
        ostr << "@soc=" << channelTile->socInd + 1 << "/" << tiling->socTiles;

    return ostr.str();
}

template <class Tiles>
std::string getPlaneTilePostfix(const HwPlaneTilePtr<Tiles>& planeTile) {
    auto tiling = planeTile->parent.lock();
    IE_ASSERT(tiling != nullptr);

    std::ostringstream ostr;

    if (tiling->sohTiles > 1)
        ostr << "@soh=" << planeTile->sohInd + 1 << "/" << tiling->sohTiles;
    if (tiling->sowTiles > 1)
        ostr << "@sow=" << planeTile->sowInd + 1 << "/" << tiling->sowTiles;

    return ostr.str();
}

struct HwConvTileInfo final {
    HwOpMode mode = HwOpMode::MODE_1_256;
    int numDescr = 0;
    int outChansPerDescr = 0;
    int lastOutChans = 0;
    int extendedInputDimC = 0;
    int extendedOutputDimC = 0;
    double cost = std::numeric_limits<double>::max();
};

void printTo(std::ostream& os, const HwConvTileInfo& convTiles);
void printTo(DotLabel& lbl, const HwConvTileInfo& convTiles);

using HwConvChannelTile = HwChannelTile<HwConvTileInfo>;
using HwConvChannelTilePtr = HwChannelTilePtr<HwConvTileInfo>;
using HwConvPlaneTile = HwPlaneTile<HwConvTileInfo>;
using HwConvPlaneTilePtr = HwPlaneTilePtr<HwConvTileInfo>;
using HwConvTiling = HwTiling<HwConvTileInfo>;
using HwConvTilingPtr = HwTilingPtr<HwConvTileInfo>;

struct HwPoolTileInfo final {
    HwOpMode mode = HwOpMode::MODE_1_256;
    int numDescr = 0;
    int chansPerDescr = 0;
};

void printTo(std::ostream& os, const HwPoolTileInfo& poolTiles);
void printTo(DotLabel& lbl, const HwPoolTileInfo& poolTiles);

using HwPoolChannelTile = HwChannelTile<HwPoolTileInfo>;
using HwPoolChannelTilePtr = HwChannelTilePtr<HwPoolTileInfo>;
using HwPoolPlaneTile = HwPlaneTile<HwPoolTileInfo>;
using HwPoolPlaneTilePtr = HwPlaneTilePtr<HwPoolTileInfo>;
using HwPoolTiling = HwTiling<HwPoolTileInfo>;
using HwPoolTilingPtr = HwTilingPtr<HwPoolTileInfo>;

struct HwFullyConnectedTileInfo final {
    HwOpMode mode = HwOpMode::MODE_1_256;
    int numOutTiles = 0;
    int numInSubTiles = 0;
    int workInN = 0;
    int workOutN = 0;
};

void printTo(std::ostream& os, const HwFullyConnectedTileInfo& fcTiles);
void printTo(DotLabel& lbl, const HwFullyConnectedTileInfo& fcTiles);

using HwFullyConnectedChannelTile = HwChannelTile<HwFullyConnectedTileInfo>;
using HwFullyConnectedChannelTilePtr = HwChannelTilePtr<HwFullyConnectedTileInfo>;
using HwFullyConnectedPlaneTile = HwPlaneTile<HwFullyConnectedTileInfo>;
using HwFullyConnectedPlaneTilePtr = HwPlaneTilePtr<HwFullyConnectedTileInfo>;
using HwFullyConnectedTiling = HwTiling<HwFullyConnectedTileInfo>;
using HwFullyConnectedTilingPtr = HwTilingPtr<HwFullyConnectedTileInfo>;

//
// Input<->Output tile calculation
//

int calcOutputSize(
        int inputSize,
        int kernelSize, int kernelStride,
        int padBefore, int padAfter,
        bool useCeil);

//
// Plane tiles calculation.
//

SmallVector<HwPlaneTileInfo> splitIntoPlaneTilesWithPool(
        int inputSize,
        int kernelSize, int kernelStride,
        int pad,
        int maxOutputSize);

SmallVector<HwPlaneTileInfo> splitIntoPlaneTiles(
        int inputSize, int outputSize,
        int kernelSize, int kernelStride,
        int padBefore, int padAfter,
        int maxOutputSize,
        bool alignInputTile,
        bool useCeil);

//
// HW Convolution tiling over output channels.
//

// This function tries to split the output over channels.
HwConvTileInfo splitHwConvIntoOutChannelsTiles(
        int inTileWidth, int inTileHeight, int inTileChannels,
        int outTileChannels,
        int kernelSizeX, int kernelSizeY,
        int kernelStride);

}  // namespace vpu
