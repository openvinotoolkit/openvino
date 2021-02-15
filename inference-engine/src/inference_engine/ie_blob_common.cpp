// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_blob.h"

#include <memory>
#include <utility>
#include <vector>

namespace InferenceEngine {

Blob::Ptr Blob::createROI(const ROI&) const {
    THROW_IE_EXCEPTION << "[NOT_IMPLEMENTED] createROI is not implemented for current type of Blob";
}

Blob::Ptr make_shared_blob(const Blob::Ptr& inputBlob, const ROI& roi) {
    return inputBlob->createROI(roi);
}

}  // namespace InferenceEngine
