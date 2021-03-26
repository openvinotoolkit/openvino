// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include "common/permute_utils.h"

using namespace InferenceEngine;

namespace MKLDNNPlugin {

class MKLDNNShuffleChannelsNode : public MKLDNNNode, PermuteUtils {
public:
    MKLDNNShuffleChannelsNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNShuffleChannelsNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    void executeRef(const float* srcData, float* dstData);

    SizeVector dataDims;
    int dataRank;
    int axis;
    size_t group;
    size_t groupSize;
};

}  // namespace MKLDNNPlugin
