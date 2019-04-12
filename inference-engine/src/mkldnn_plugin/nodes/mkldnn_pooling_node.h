// Copyright (C) 2018-2019 Intel Corporation
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
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    std::vector<ptrdiff_t> kernel;
};

}  // namespace MKLDNNPlugin

