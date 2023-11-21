// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_input_info.hpp"

#include "cpu_shape.h"
#include "ie_parallel.hpp"
#include <vector>
#include <limits>

namespace ov {
namespace intel_cpu {

class NormalizePreprocess {
public:
    NormalizePreprocess();

public:
    void Load(const Shape& inputShape, InferenceEngine::InputInfo::Ptr inputInfo);
    void NormalizeImage(const Shape &inputShape, float *input, InferenceEngine::Layout layout);

    template<typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
    void NormalizeImage(const Shape &inputShape, T *input, InferenceEngine::Layout layout) {
        OPENVINO_ASSERT(input != nullptr);

        const auto inputDims = inputShape.getStaticDims();
        if (inputDims.size() != 4) {
            OPENVINO_THROW("Expecting input as 4 dimension blob with format NxCxHxW.");
        }

        if (layout != InferenceEngine::NCHW && layout != InferenceEngine::NHWC) {
            OPENVINO_THROW("Expecting input layout NCHW or NHWC.");
        }

        int MB = inputDims[0];
        int srcSize = inputShape.getElementsCount() / MB;

        if (meanBuffer && meanBuffer->size()) {
            const float * meanBufferValues = meanBuffer->readOnly();

            InferenceEngine::parallel_for2d(MB, srcSize, [&](int mb, int i) {
                int buf = input[srcSize * mb + i];
                buf -= meanBufferValues[i];
                if (buf < (std::numeric_limits<T>::min)()) buf = (std::numeric_limits<T>::min)();
                if (buf > (std::numeric_limits<T>::max)()) buf = (std::numeric_limits<T>::max)();
                input[srcSize * mb + i] = buf;
            });
        } else if (!meanValues.empty() && !stdScales.empty()) {
            int C = inputDims[1];
            srcSize /= inputDims[1];

            for (int c = 0; c < C; c++) {
                if (stdScales[c] != 1)
                    OPENVINO_THROW("Preprocessing error: fractional normalization is not supported for integer data. ");
            }

            if (layout == InferenceEngine::NCHW) {
                InferenceEngine::parallel_for3d(MB, C, srcSize, [&](int mb, int c, int i) {
                    int buf = input[srcSize * mb * C + c * srcSize + i];
                    buf -= meanValues[c];
                    if (buf < (std::numeric_limits<T>::min)()) buf = (std::numeric_limits<T>::min)();
                    if (buf > (std::numeric_limits<T>::max)()) buf = (std::numeric_limits<T>::max)();
                    input[srcSize * mb * C + c * srcSize + i] = buf;
                });
            } else if (layout == InferenceEngine::NHWC) {
                InferenceEngine::parallel_for2d(MB, srcSize, [&](int mb, int i) {
                    for (int c = 0; c < C; c++) {
                        int buf = input[mb * srcSize * C + i * C + c];
                        buf -= meanValues[c];
                        if (buf < (std::numeric_limits<T>::min)()) buf = (std::numeric_limits<T>::min)();
                        if (buf > (std::numeric_limits<T>::max)()) buf = (std::numeric_limits<T>::max)();
                        input[mb * srcSize * C + i * C + c] = buf;
                    }
                });
            }
        } else {
            OPENVINO_THROW("Preprocessing error: meanValues and stdScales arrays are inconsistent.");
        }
    }

private:
    std::vector<float> meanValues;

    std::vector<float> stdScales;

    InferenceEngine::TBlob<float>::Ptr meanBuffer;
};

}   // namespace intel_cpu
}   // namespace ov
