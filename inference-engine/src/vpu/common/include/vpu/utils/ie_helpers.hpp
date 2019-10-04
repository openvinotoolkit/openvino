// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(LayoutPreference,
    AUTO,
    ChannelMajor,  // CHW, NCHW, NCDHW
    ChannelMinor   // HWC, NHWC, NDHWC
)

InferenceEngine::Layout deviceLayout(InferenceEngine::Layout const& layout,
                                       vpu::LayoutPreference const& layoutPreference);

ie::Blob::Ptr getBlobFP16(const ie::Blob::Ptr& in);

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in, ie::Layout outLayout);
void copyBlob(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out);

}  // namespace vpu
