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
#include "mkldnn_memory.h"

namespace MKLDNNPlugin {

class MemoryDescUtils {
public:
    static mkldnn::memory::format_tag getLayout(const MemoryDesc& desc);

    static InferenceEngine::TensorDesc convertToTensorDesc(const MemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const MemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const BlockedMemoryDesc& desc);
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const InferenceEngine::TensorDesc& desc);

    static MKLDNNMemoryDesc getUndefinedMemoryDesc(const MKLDNNMemoryDesc& desc);
};

}  // namespace MKLDNNPlugin
