// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNSoftMaxNode : public MKLDNNNode {
public:
    MKLDNNSoftMaxNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNSoftMaxNode() override = default;

    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

private:
    static Register<MKLDNNSoftMaxNode> reg;
    int axis = 0;
};

}  // namespace MKLDNNPlugin

