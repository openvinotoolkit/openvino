// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_preprocess.h"
#include "ie_parallel.hpp"
#include "nodes/common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

NormalizePreprocess::NormalizePreprocess() : meanBuffer(nullptr) {
}

void NormalizePreprocess::Load(const MKLDNNDims& inputDims, InputInfo::Ptr inputInfo) {
    PreProcessInfo &pp = inputInfo->getPreProcess();
    size_t inChannels = pp.getNumberOfChannels();
    if (inChannels == 0) {
        meanBuffer = nullptr;
        return;
    }

    if (inChannels != inputDims[1]) {
        IE_THROW() << "channels mismatch between mean and input";
    }

    switch (pp.getMeanVariant()) {
        case MEAN_VALUE: {
            // mean and standard deviation image common value per channel (1x1xC)
            meanValues.resize(inChannels);
            stdScales.resize(inChannels);

            for (unsigned channel = 0; channel < inChannels; channel++) {
                if (pp[channel]->stdScale == 0) {
                    IE_THROW() << "Preprocessing error: stdScale cannot be equal zero";
                }
                meanValues[channel] = pp[channel]->meanValue;
                stdScales[channel] = pp[channel]->stdScale;
            }
        }
        break;
        case MEAN_IMAGE: {
            // since MKLDNN expects all channels in the same buffer - we copy it here as it comes from different channels...
            auto meanWidth = pp[0]->meanData->getTensorDesc().getDims()[pp[0]->meanData->getTensorDesc().getDims().size() - 1];
            auto meanHeight = pp[0]->meanData->getTensorDesc().getDims()[pp[0]->meanData->getTensorDesc().getDims().size() - 2];

            TensorDesc desc(Precision::FP32, {inChannels, meanHeight, meanWidth}, Layout::CHW);

            meanBuffer = make_shared_blob<float>(desc);

            meanBuffer->allocate();

            for (unsigned channel = 0; channel < inChannels; channel++) {
                Blob::Ptr meanBlob = pp[channel]->meanData;
                if (!meanBlob || meanBlob->getTensorDesc().getPrecision() != Precision::FP32)
                    IE_THROW() << "mean image not provided or not in Float 32";
                if (meanBlob->size() != meanHeight*meanWidth) {
                    IE_THROW() << "mean image size does not match expected network input, expecting " << meanWidth << " x " << meanHeight;
                }
                // todo: cast to TBlob and make sure it is floats
                cpu_memcpy_s(meanBuffer->data() + channel*meanBlob->size(), meanBuffer->byteSize() - channel*meanBlob->byteSize(),
                          meanBlob->buffer(), meanBlob->byteSize());
            }
        }
            break;

        case NONE: {
            // there is no mean image. So disable mean image step
            meanBuffer = nullptr;
        }
            break;

        default: {
            IE_THROW() << "Unsupported mean variant: " << pp.getMeanVariant();
        }
    }
}

void NormalizePreprocess::NormalizeImage(const MKLDNNDims &inputDims, float *input, InferenceEngine::Layout layout) {
    IE_ASSERT(input != nullptr);

    if (inputDims.ndims() != 4) {
        IE_THROW() << "Expecting input as 4 dimension blob with format NxCxHxW.";
    }

    if (layout != NCHW && layout != NHWC) {
        IE_THROW() << "Expecting input layout NCHW or NHWC.";
    }

    int MB = inputDims[0];
    int srcSize = inputDims.size() / MB;

    if (meanBuffer && meanBuffer->size()) {
        const float * meanBufferValues = meanBuffer->readOnly();

        parallel_for2d(MB, srcSize, [&](int mb, int i) {
            input[srcSize * mb + i] -= meanBufferValues[i];
        });
    } else if (!meanValues.empty() && !stdScales.empty()) {
        int C = inputDims[1];
        srcSize /= inputDims[1];

        if (layout == NCHW) {
            parallel_for3d(MB, C, srcSize, [&](int mb, int c, int i) {
                input[mb * C * srcSize + c * srcSize + i] -= meanValues[c];
                input[mb * C * srcSize + c * srcSize + i] /= stdScales[c];
            });
        } else if (layout == NHWC) {
            parallel_for2d(MB, srcSize, [&](int mb, int i) {
                for (int c = 0; c < C; c++) {
                    input[mb * srcSize * C + i * C + c] -= meanValues[c];
                    input[mb * srcSize * C + i * C + c] /= stdScales[c];
                }
            });
        }
    } else {
        IE_THROW() << "Preprocessing error: meanValues and stdScales arrays are inconsistent.";
    }
}
