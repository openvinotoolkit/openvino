// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace preprocess {

enum class ResizeAlgorithm {
    RESIZE_LINEAR,
    RESIZE_CUBIC,
    RESIZE_NEAREST,
    RESIZE_BILINEAR_PILLOW,
    RESIZE_BICUBIC_PILLOW
};

}  // namespace preprocess
}  // namespace ov
