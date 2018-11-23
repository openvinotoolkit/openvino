// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

enum class ResizeCropPolicy {
    DoNothing,
    Resize,
    ResizeThenCrop,
};

struct PreprocessingOptions {
    // Normal image channel values are 1 byte (0..255).
    // But some topologies (i.e. YOLO) input values scaled to 0..1
    bool scaleValuesTo01;

    ResizeCropPolicy resizeCropPolicy;

    // If resizeCropPolicy is ResizeThenCrop, these variables contain
    // the size before cropping
    size_t resizeBeforeCropX, resizeBeforeCropY;

    PreprocessingOptions() : scaleValuesTo01(false), resizeCropPolicy(ResizeCropPolicy::DoNothing), resizeBeforeCropX(0), resizeBeforeCropY(0) { }

    PreprocessingOptions(bool scaleValuesTo01, ResizeCropPolicy resizeCropPolicy, size_t resizeBeforeCropX = 0, size_t resizeBeforeCropY = 0)
        : scaleValuesTo01(scaleValuesTo01), resizeCropPolicy(resizeCropPolicy), resizeBeforeCropX(resizeBeforeCropX), resizeBeforeCropY(resizeBeforeCropY) {
    }
};
