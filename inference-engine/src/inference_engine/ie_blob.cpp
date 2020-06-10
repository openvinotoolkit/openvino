// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "ie_blob.h"

namespace InferenceEngine {

Blob::Blob(const Blob& blob): tensorDesc(blob.tensorDesc), roiPtr(blob.roiPtr ? new ROIData(*blob.roiPtr) : nullptr) {}

void Blob::setROI(const ROI& roi) {
    TensorDesc origTensorDesc = roiPtr ? roiPtr->original : tensorDesc;
    roiPtr = std::unique_ptr<ROIData>(new ROIData({roi, origTensorDesc}));
    tensorDesc = make_roi_tensor_desc(origTensorDesc, roi);
}

Blob::Ptr Blob::CreateROIBlob(const ROI& roi) const {
    Blob* roiBlob = this->clone();
    try {
        roiBlob->setROI(roi);
        return std::shared_ptr<Blob>(roiBlob);
    } catch (const std::exception& ex) {
        delete roiBlob;
        THROW_IE_EXCEPTION << "Cannot create ROI blob with the specified parameters: " << ex.what();
    }
}

}  // namespace InferenceEngine
