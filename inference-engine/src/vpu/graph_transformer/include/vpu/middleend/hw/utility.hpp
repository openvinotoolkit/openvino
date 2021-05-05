// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <ostream>

#include <vpu/model/data.hpp>
#include <vpu/backend/blob_format.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

//
// HW Operation parameters
//

VPU_DECLARE_ENUM(HwOpType,
    CONV = 0,
    CONV_POOL = 1,
    FC = 2,
    POOL = 4,
)

VPU_DECLARE_ENUM(HwPoolType,
    MAX = 0,
    AVERAGE = 1,
);

VPU_DECLARE_ENUM(HwOpMode,
    MODE_1_256 = 0,
    MODE_2_128 = 1,
    MODE_4_64 = 2,
    MODE_8_32 = 3,
    MODE_16_16 = 4,
);

VPU_DECLARE_ENUM(HwPadMode,
    PAD_WITH_ZEROS = 0x00,
    PAD_REPEAT_RIGHT_EDGE = 0x01,
    PAD_REPEAT_LEFT_EDGE = 0x08,
    PAD_REPEAT_TOP_EDGE = 0x04,
    PAD_REPEAT_BOTTOM_EDGE = 0x02,
);

inline HwPadMode operator|(HwPadMode m1, HwPadMode m2) {
    return static_cast<HwPadMode>(static_cast<int32_t>(m1) | static_cast<int32_t>(m2));
}

VPU_DECLARE_ENUM(HwCoeffMode,
    FP16 = 0,
    U8F = 1,
);

VPU_DECLARE_ENUM(HwDataMode,
    FP16 = 0,
    U8F = 1,
);

struct HwOpParams final {
    HwOpType opType = HwOpType::CONV;
    HwOpMode opMode = HwOpMode::MODE_1_256;

    HwPoolType poolType = HwPoolType::MAX;

    bool withPad = false;
    HwPadMode padMode = HwPadMode::PAD_WITH_ZEROS;

    int32_t inputInd = -1;
    int32_t outputInd = -1;
    int32_t coeffsInd = -1;
    int32_t biasesInd = -1;
    int32_t scalesInd = -1;

    uint32_t outChanOffset = 0;
    uint32_t outNumChans = 0;

    uint32_t fcInputOffset = 0;
    uint32_t fcInputNum = 0;
    uint32_t fcOutputOffset = 0;
    uint32_t fcOutputNum = 0;
    bool fcAccum = false;

    uint32_t kernelWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelStride = 0;

    uint32_t poolKernelWidth = 0;
    uint32_t poolKernelHeight = 0;

    bool withReLU = false;
    uint32_t t0 = 0;
    uint32_t a0 = 0;
    uint32_t a1 = 0;

    bool withClamp = false;
    float clampMaxVal = 0;

    bool reuseData = false;
    bool reuseCoeff = false;
};

struct HwOpList final {
    SmallVector<HwOpParams> vec;
};

void printTo(std::ostream& os, const HwOpList& hwOps);
void printTo(DotLabel& lbl, const HwOpList& hwOps);

//
// HwPaddingInfo
//

struct HwPaddingInfo final {
    bool enable = false;
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;
};

HwPaddingInfo getHwPaddingInfo(
        const DimValues& inDims, const DimValues& outDims,
        int kernelDimX, int kernelDimY,
        int kernelStrideX, int kernelStrideY,
        int padLeft, int padTop);

void printTo(std::ostream& os, const HwPaddingInfo& hwPad);
void printTo(DotLabel& lbl, const HwPaddingInfo& hwPad);

//
// calculateHwBufferSize
//

int calculateHwBufferSize(const DimValues& dims, const DimsOrder& order = DimsOrder());

}  // namespace vpu
