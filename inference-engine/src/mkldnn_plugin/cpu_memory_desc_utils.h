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

class MemoryDescUtils {
public:
    static InferenceEngine::TensorDesc convertToTensorDesc(const MemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const MemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const BlockedMemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const InferenceEngine::TensorDesc& desc);

private:
    static BlockedMemoryDesc convertToBlockedDescriptor(const MKLDNNMemoryDesc& inpDesc);
    static BlockedMemoryDesc convertToBlockedDescriptor(const MemoryDesc& desc);

    friend class MKLDNNMemory;
    friend class MKLDNNGraphOptimizer;

    //static MemoryDescPtr getUndefinedMemoryDesc(const MKLDNNMemoryDesc& desc);
};

}  // namespace MKLDNNPlugin
