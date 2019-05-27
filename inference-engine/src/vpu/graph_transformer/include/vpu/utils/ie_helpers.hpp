// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>

namespace vpu {

namespace ie = InferenceEngine;

ie::Blob::Ptr getBlobFP16(const ie::Blob::Ptr& in);

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in);
ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in, ie::Layout outLayout);
void copyBlob(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out);

}  // namespace vpu
