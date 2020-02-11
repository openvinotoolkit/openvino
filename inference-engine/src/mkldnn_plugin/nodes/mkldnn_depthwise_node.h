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

class MKLDNNDepthwiseNode : public MKLDNNNode {
public:
    MKLDNNDepthwiseNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNDepthwiseNode() override = default;

    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void initOptimalPrimitiveDescriptor() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    mkldnn::algorithm getAlgorithm() {
        if (!initialized)
            initValues();
        return algorithm;
    }

    bool isWithBiases() {
        if (!initialized)
            initValues();
        return withBiases;
    }

    bool isBroadcast() {
        if (!initialized)
            initValues();
        return broadcast;
    }

private:
    void initValues();
    bool initialized = false;

    mkldnn::algorithm algorithm = mkldnn::algorithm::depthwise_scale_shift;
    size_t realWeightSize = 0;
    size_t realBiasSize = 0;
    bool withBiases = false;
    bool broadcast = false;
};

}  // namespace MKLDNNPlugin
