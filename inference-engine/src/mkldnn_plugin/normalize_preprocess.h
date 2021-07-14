// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_input_info.hpp"

#include "cpu_shape.h"
#include "ie_parallel.hpp"
#include <vector>
#include <limits>

namespace MKLDNNPlugin {

class NormalizePreprocess {
public:
    NormalizePreprocess();

public:
    void Load(const Shape& inputShape, InferenceEngine::InputInfo::Ptr inputInfo);
    void NormalizeImage(const MKLDNNDims &inputDims, float *input, InferenceEngine::Layout layout);

    template<typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
    void NormalizeImage(const MKLDNNDims &inputDims, T *input, InferenceEngine::Layout layout) {
        IE_ASSERT(input != nullptr);

        if (inputDims.ndims() != 4) {
            IE_THROW() << "Expecting input as 4 dimension blob with format NxCxHxW.";
        }

        if (layout != InferenceEngine::NCHW && layout != InferenceEngine::NHWC) {
            IE_THROW() << "Expecting input layout NCHW or NHWC.";
        }

        int MB = inputDims[0];
        int srcSize = inputDims.size() / MB;

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
                    IE_THROW() << "Preprocessing error: fractional normalization is not supported for integer data. ";
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
            IE_THROW() << "Preprocessing error: meanValues and stdScales arrays are inconsistent.";
        }
    }

private:
    std::vector<float> meanValues;

    std::vector<float> stdScales;

    InferenceEngine::TBlob<float>::Ptr meanBuffer;
};

}  // namespace MKLDNNPlugin
