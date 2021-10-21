// Copyright (C) 2018-2021 Intel Corporation
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
    RGB,
    BGR
};

}  // namespace preprocess
}  // namespace ov
