// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>
#include <vector>

#include "blob_factory.hpp"
#include "ie_blob.h"

namespace InferenceEngine {

TensorDesc make_roi_tensor_desc(const TensorDesc& inputTensorDesc, const ROI& roi) {
    const size_t blkDimsH = roi.sizeY;
    const size_t blkDimsW = roi.sizeX;
    const size_t blkDimsC = inputTensorDesc.getDims()[1];
    size_t blkOffset;
    SizeVector blkOrder;
    SizeVector blkDims;

    if (roi.posX + roi.sizeX > inputTensorDesc.getDims()[3] || roi.posY + roi.sizeY > inputTensorDesc.getDims()[2]) {
        THROW_IE_EXCEPTION << "passed ROI coordinates are inconsistent to input size";
    }

    Layout blobLayout = inputTensorDesc.getLayout();
    switch (blobLayout) {
    case NCHW: {
        blkOffset = inputTensorDesc.getDims()[3] * roi.posY + roi.posX;
        blkOrder = {0, 1, 2, 3};
        blkDims = {1, blkDimsC, blkDimsH, blkDimsW};  // we use BlockingDesc for 1 cropped image only
    } break;
    case NHWC: {
        blkOffset = blkDimsC * (inputTensorDesc.getDims()[3] * roi.posY + roi.posX);
        blkOrder = {0, 2, 3, 1};
        blkDims = {1, blkDimsH, blkDimsW, blkDimsC};  // we use BlockingDesc for 1 cropped image only
    } break;
    default: {
        THROW_IE_EXCEPTION << "ROI could not be cropped due to unsupported input layout: " << blobLayout;
    }
    }

    // the strides are the same because ROI blob uses the same memory buffer as original input blob.
    SizeVector blkStrides(inputTensorDesc.getBlockingDesc().getStrides());

    SizeVector blkDimsOffsets = {0, 0, 0, 0};  // no offset per dims by default

    BlockingDesc blkDesc(blkDims, blkOrder, blkOffset, blkDimsOffsets, blkStrides);
    TensorDesc outputTensorDesc(inputTensorDesc.getPrecision(), {1, blkDimsC, blkDimsH, blkDimsW}, blkDesc);
    outputTensorDesc.setLayout(blobLayout);

    return outputTensorDesc;
}

Blob::Ptr make_shared_blob(const Blob::Ptr& inputBlob, const ROI& roi) {
    return inputBlob->CreateROIBlob(roi);
}

}  // namespace InferenceEngine
