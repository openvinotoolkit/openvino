// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>
#include <vector>

#include "ie_blob.h"

namespace InferenceEngine {

Blob::Ptr Blob::createROI(const ROI&) const {
    IE_THROW(NotImplemented) << "createROI is not implemented for current type of Blob";
}

Blob::Ptr make_shared_blob(const Blob::Ptr& inputBlob, const ROI& roi) {
    return inputBlob->createROI(roi);
}

}  // namespace InferenceEngine
