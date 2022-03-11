// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/utils/enums.hpp>
#include <vpu/utils/io.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/error.hpp>

#include <ie_data.h>
#include <ie_blob.h>
#include <legacy/ie_layers.h>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(LayoutPreference,
    ChannelMajor,  // CHW, NCHW, NCDHW
    ChannelMinor   // HWC, NHWC, NDHWC
)

InferenceEngine::Layout deviceLayout(InferenceEngine::Layout const& layout,
                                     LayoutPreference const& layoutPreference);

ie::Blob::Ptr convertBlobFP32toFP16(const ie::Blob::CPtr& in);

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& original);
ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in, ie::Layout outLayout, void* ptr = nullptr);
void copyBlob(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out);

void printTo(DotLabel& lbl, const ie::DataPtr& ieData);
void printTo(DotLabel& lbl, const ie::Blob::Ptr& ieBlob);
void printTo(DotLabel& lbl, const ie::CNNLayerPtr& ieLayer);

}  // namespace vpu
