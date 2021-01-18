// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include "common/permute_utils.h"

namespace MKLDNNPlugin {

class MKLDNNSpaceToDepthNode : public MKLDNNNode, PermuteUtils {
public:
    MKLDNNSpaceToDepthNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNSpaceToDepthNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    enum SpaceToDepthMode {
        BLOCKS_FIRST = 0,
        DEPTH_FIRST = 1
    };

    SpaceToDepthMode mode;
    size_t blockSize;
    size_t blockStep;

    struct {
        InferenceEngine::SizeVector shape5D;
        InferenceEngine::SizeVector block3D;
        size_t spatialStep;
        size_t batchStep;
        size_t srcChannels;
        size_t dstChannels;
        size_t blockShift;
        size_t channelShift;
        InferenceEngine::SizeVector dst_strides;
        InferenceEngine::SizeVector src_strides;
    } params;
};

}  // namespace MKLDNNPlugin
