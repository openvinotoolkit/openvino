// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ie_api.h>

namespace InferenceEngine {

typedef short ie_fp16;

namespace PrecisionUtils {

INFERENCE_ENGINE_API_CPP(ie_fp16) f32tof16(float x);

INFERENCE_ENGINE_API_CPP(float) f16tof32(ie_fp16 x);

INFERENCE_ENGINE_API_CPP(void) f16tof32Arrays(float *dst, const short *src, size_t nelem, float scale = 1.f, float bias = 0.f);

INFERENCE_ENGINE_API_CPP(void) f32tof16Arrays(short *dst, const float *src, size_t nelem, float scale = 1.f, float bias = 0.f);

}  // namespace PrecisionUtils

}  // namespace InferenceEngine
