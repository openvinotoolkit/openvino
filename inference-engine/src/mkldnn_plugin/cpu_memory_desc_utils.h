// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "perf_count.h"
#include <vector>
#include <utility>
#include <ie_common.h>
#include <ie_layouts.h>
#include "mkldnn_dims.h"

namespace MKLDNNPlugin {
class MKLDNNMemoryDesc;
class BlockedMemoryDesc;
class MKLDNNMemory;

class MemoryDescUtils {
public:
    static InferenceEngine::TensorDesc convertToTensorDesc(const MemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const MemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const BlockedMemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const InferenceEngine::TensorDesc& desc);
    static BlockedMemoryDesc convertToBlockedDescriptor(const MemoryDesc& desc);
    static BlockedMemoryDesc convertToBlockedDescriptor(const MKLDNNMemoryDesc& inpDesc);
    static MemoryDescPtr applyUndefinedOffset(const MKLDNNMemoryDesc& desc);
    static MemoryDescPtr applyUndefinedOffset(const BlockedMemoryDesc& desc);
    static MemoryDescPtr resetOffset(const MemoryDesc* desc);
    static InferenceEngine::Blob::Ptr interpretAsBlob(const MKLDNNMemory& mem);
};

}  // namespace MKLDNNPlugin
