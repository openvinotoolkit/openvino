// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace preprocess {

/// \brief An enum containing all supported padding modes in preprocessing
enum class PaddingMode {
    PAD_CONSTANT,           //!< padded values are taken from the value/values input
    PAD_EDGE,               //!< Padded values are copied from the respective edge of the input data tensor
    PAD_REFLECT,            //!< Padded values are computed by reflecting the input data tensor at the edge. Values on the edges are not duplicated.
    PAD_SYMMETRIC           //!< Padded values are computed by reflecting the input data tensor at the edge, but values on the edges are duplicated.
};

}  // namespace preprocess
}  // namespace ov
