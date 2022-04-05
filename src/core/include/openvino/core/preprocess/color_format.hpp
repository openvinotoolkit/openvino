// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace preprocess {

/// \brief Color format enumeration for conversion
enum class ColorFormat {
    UNDEFINED,
    NV12_SINGLE_PLANE,  // Image in NV12 format as single tensor
    /// \brief Image in NV12 format represented as separate tensors for Y and UV planes.
    NV12_TWO_PLANES,
    I420_SINGLE_PLANE,  // Image in I420 (YUV) format as single tensor
    /// \brief Image in I420 format represented as separate tensors for Y, U and V planes.
    I420_THREE_PLANES,
    RGB,
    BGR,
    /// \brief Image in RGBX interleaved format (4 channels)
    RGBX,
    /// \brief Image in BGRX interleaved format (4 channels)
    BGRX
};

}  // namespace preprocess
}  // namespace ov
