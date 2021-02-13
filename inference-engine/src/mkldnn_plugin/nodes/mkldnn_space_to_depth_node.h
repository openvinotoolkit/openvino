// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include "common/depth_and_space_utils.h"

namespace MKLDNNPlugin {

class MKLDNNSpaceToDepthNode : public MKLDNNNode, DepthAndSpaceUtils {
public:
    MKLDNNSpaceToDepthNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNSpaceToDepthNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
};

}  // namespace MKLDNNPlugin
