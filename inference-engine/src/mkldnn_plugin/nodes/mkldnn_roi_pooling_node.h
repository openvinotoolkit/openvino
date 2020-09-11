// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNROIPoolingNode : public MKLDNNNode {
public:
    MKLDNNROIPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNROIPoolingNode() override = default;

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void createPrimitive() override;
    bool created() const override;

private:
    int pooled_h = 0;
    int pooled_w = 0;
    float spatial_scale = 0;
    mkldnn::algorithm method = mkldnn::algorithm::roi_pooling_max;
};

}  // namespace MKLDNNPlugin

