// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"

namespace ov {
namespace preprocess {

/// \brief Flip mode for preprocessing
enum class FlipMode {
    HORIZONTAL,  ///< Flip along the Width axis (Mirror)
    VERTICAL     ///< Flip along the Height axis (Upside down)
};

}  // namespace preprocess
}  // namespace ov