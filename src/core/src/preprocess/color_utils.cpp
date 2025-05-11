// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "color_utils.hpp"

using namespace ov::preprocess;

std::unique_ptr<ColorFormatInfo> ColorFormatInfo::get(ColorFormat format) {
    std::unique_ptr<ColorFormatInfo> res;
    switch (format) {
    case ColorFormat::NV12_SINGLE_PLANE:
    case ColorFormat::I420_SINGLE_PLANE:
        res.reset(new ColorFormatInfoYUV420_Single(format));
        break;
    case ColorFormat::NV12_TWO_PLANES:
        res.reset(new ColorFormatInfoNV12_TwoPlanes(format));
        break;
    case ColorFormat::I420_THREE_PLANES:
        res.reset(new ColorFormatInfoI420_ThreePlanes(format));
        break;
    case ColorFormat::RGB:
        res.reset(new ColorFormatRGB(format));
        break;
    case ColorFormat::BGR:
        res.reset(new ColorFormatBGR(format));
        break;
    case ColorFormat::RGBX:
    case ColorFormat::BGRX:
        res.reset(new ColorFormatInfo_RGBX_Base(format));
        break;
    case ColorFormat::GRAY:
        res.reset(new ColorFormatNHWC(format));
        break;
    default:
        res.reset(new ColorFormatInfo(format));
        break;
    }
    return res;
}
