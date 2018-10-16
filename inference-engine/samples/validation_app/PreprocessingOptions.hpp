/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
