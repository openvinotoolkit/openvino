// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_blob.h"
#include "ie_compound_blob.h"
#include "blob_factory.hpp"

#include <memory>
#include <vector>
#include <utility>

namespace InferenceEngine {

Blob::Ptr make_shared_blob(const Blob::Ptr &inputBlob, const ROI &roi) {
    // reject compound blobs
    if (inputBlob->is<CompoundBlob>()) {
        THROW_IE_EXCEPTION << "Compound blobs do not support ROI";
    }

    size_t blkDimsH = roi.sizeY;
    size_t blkDimsW = roi.sizeX;
    size_t blkDimsC = inputBlob->getTensorDesc().getDims()[1];
    size_t blkOffset;
    SizeVector blkOrder;
    SizeVector blkDims;

    if (roi.posX + roi.sizeX > inputBlob->getTensorDesc().getDims()[3] ||
        roi.posY + roi.sizeY > inputBlob->getTensorDesc().getDims()[2]) {
        THROW_IE_EXCEPTION << "passed ROI coordinates are inconsistent to input size";
    }

    Layout blobLayout = inputBlob->getTensorDesc().getLayout();
    switch (blobLayout) {
        case NCHW: {
            blkOffset = inputBlob->getTensorDesc().getDims()[3] * roi.posY + roi.posX;
            blkOrder = {0, 1, 2, 3};
            blkDims = {1, blkDimsC, blkDimsH, blkDimsW};  // we use BlockingDesc for 1 cropped image only
        }
        break;
        case NHWC: {
            blkOffset = blkDimsC * (inputBlob->getTensorDesc().getDims()[3] * roi.posY + roi.posX);
            blkOrder = {0, 2, 3, 1};
            blkDims = {1, blkDimsH, blkDimsW, blkDimsC};  // we use BlockingDesc for 1 cropped image only
        }
        break;
        default: {
            THROW_IE_EXCEPTION << "ROI could not be cropped due to inconsistent input layout: " << blobLayout;
        }
    }

    // the strides are the same because ROI blob uses the same memory buffer as original input blob.
    SizeVector blkStrides(inputBlob->getTensorDesc().getBlockingDesc().getStrides());

    SizeVector blkDimsOffsets = {0, 0, 0, 0};  // no offset per dims by default

    BlockingDesc blkDesc(blkDims, blkOrder, blkOffset, blkDimsOffsets, blkStrides);
    TensorDesc tDesc(inputBlob->getTensorDesc().getPrecision(), {1, blkDimsC, blkDimsH, blkDimsW}, blkDesc);
    tDesc.setLayout(blobLayout);

    return make_blob_with_precision(tDesc, inputBlob->buffer());
}

}  // namespace InferenceEngine
