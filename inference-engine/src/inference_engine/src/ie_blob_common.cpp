// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>
#include <vector>

#include "ie_blob.h"

namespace InferenceEngine {
Blob::Ptr Blob::createROI(const ROI& roi) const {
    if (getTensorDesc().getLayout() == Layout::NCHW || getTensorDesc().getLayout() == Layout::NHWC) {
        return createROI({roi.id, 0, roi.posY, roi.posX},
                         {roi.id + 1, getTensorDesc().getDims()[1], roi.posY + roi.sizeY, roi.posX + roi.sizeX});
    }
    IE_THROW(NotImplemented) << "createROI is not implemented for current type of Blob";
}

Blob::Ptr Blob::createROI(const std::vector<std::size_t>& begin, const std::vector<std::size_t>& end) const {
    IE_THROW(NotImplemented) << "createROI is not implemented for current type of Blob or roi";
}

Blob::Ptr make_shared_blob(const Blob::Ptr& inputBlob, const ROI& roi) {
    return inputBlob->createROI(roi);
}

Blob::Ptr make_shared_blob(const Blob::Ptr& inputBlob,
                           const std::vector<std::size_t>& begin,
                           const std::vector<std::size_t>& end) {
    return inputBlob->createROI(begin, end);
}

}  // namespace InferenceEngine
