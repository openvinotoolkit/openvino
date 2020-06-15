// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_memory.h"

// TODO: remove this include after clean plugin code from IE::LayerConfig usage
#include "ie_extension.h"

namespace MKLDNNPlugin {

class MKLDNNPortConfig {
public:
    /** Layout descriptor of corresponding node port */
    MKLDNNMemoryDesc desc;

    /** Index of in-place memory. If -1 memory cannot be in-place */
    int inPlace = -1;

    /** Flag describing that output tensor values depend only on input tensor shape. */
    bool constant = false;
};

/**
 * The descriptor of layout configuration for some node implementation
 * Should describe some specific feature and requirements of future
 * implementation instance if it will be selected.
 */
class MKLDNNLayoutConfig {
public:
    /** Flag describing support of dynamic batch. If false, dynamic batch is not supported */
    bool dynBatchSupport = false;

    /** Vector of input data configs */
    std::vector<MKLDNNPortConfig> inConfs;

    /** Vector of output data configs */
    std::vector<MKLDNNPortConfig> outConfs;

    // TODO: This constructor is only for compatibility with old version of initialization via IE::LayerConfig
    MKLDNNLayoutConfig(const InferenceEngine::LayerConfig &ie_layer_config);
};

}  // namespace MKLDNNPlugin
