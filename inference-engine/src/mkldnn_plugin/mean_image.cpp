// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mean_image.h"
#include "ie_parallel.hpp"
#include "ie_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MeanImage::MeanImage() : meanBuffer(nullptr) {
}

void MeanImage::Load(const MKLDNNDims& inputDims, InputInfo::Ptr inputInfo) {
    PreProcessInfo &pp = inputInfo->getPreProcess();
    size_t inChannels = pp.getNumberOfChannels();
    if (inChannels == 0) {
        meanBuffer = nullptr;
        return;
    }

    if (inChannels != inputDims[1]) {
        THROW_IE_EXCEPTION << "channels mismatch between mean and input";
    }

    switch (pp.getMeanVariant()) {
        case MEAN_VALUE: {
            // mean image common value per channel (1x1xC)
            meanValues.resize(inChannels);

            for (unsigned channel = 0; channel < inChannels; channel++) {
                meanValues[channel] = pp[channel]->meanValue;
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
                    THROW_IE_EXCEPTION << "mean image not provided or not in Float 32";
                if (meanBlob->size() != meanHeight*meanWidth) {
                    THROW_IE_EXCEPTION << "mean image size does not match expected network input, expecting " << meanWidth << " x " << meanHeight;
                }
                // todo: cast to TBlob and make sure it is floats
                ie_memcpy(meanBuffer->data() + channel*meanBlob->size(), meanBuffer->byteSize() - channel*meanBlob->byteSize(),
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
            THROW_IE_EXCEPTION << "Unsupported mean variant: " << pp.getMeanVariant();
        }
    }
}

void MeanImage::Subtract(const MKLDNNDims &inputDims, float *input, InferenceEngine::Layout layout) {
    IE_ASSERT(input != nullptr);

    if (inputDims.ndims() != 4) {
        THROW_IE_EXCEPTION << "Expecting input as 4 dimension blob with format NxCxHxW.";
    }

    if (layout != NCHW && layout != NHWC) {
        THROW_IE_EXCEPTION << "Expecting input layout NCHW or NHWC.";
    }

    int MB = inputDims[0];
    int srcSize = inputDims.size() / MB;

    if (meanBuffer && meanBuffer->size()) {
        const float * meanBufferValues = meanBuffer->readOnly();

        parallel_for2d(MB, srcSize, [&](int mb, int i) {
            input[srcSize * mb + i] -= meanBufferValues[i];
        });
    } else if (!meanValues.empty()) {
        int C = inputDims[1];
        srcSize /= inputDims[1];

        if (layout == NCHW) {
            parallel_for3d(MB, C, srcSize, [&](int mb, int c, int i) {
                input[mb * C * srcSize + c * srcSize + i] -= meanValues[c];
            });
        } else if (layout == NHWC) {
            parallel_for2d(MB, srcSize, [&](int mb, int i) {
                for (int c = 0; c < C; c++)
                    input[mb * srcSize * C + i * C + c] -= meanValues[c];
            });
        }
    }
}
