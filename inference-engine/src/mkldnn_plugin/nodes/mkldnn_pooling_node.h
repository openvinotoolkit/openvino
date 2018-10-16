// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNPoolingNode : public MKLDNNNode {
public:
    MKLDNNPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNPoolingNode() override = default;

    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    static Register<MKLDNNPoolingNode> reg;
    InferenceEngine::PoolingLayer::PoolType type;
    bool exclude_pad;
    std::vector<int> stride;
    std::vector<int> paddingL;
    std::vector<int> paddingR;
    std::vector<int> kernel;
};

}  // namespace MKLDNNPlugin

