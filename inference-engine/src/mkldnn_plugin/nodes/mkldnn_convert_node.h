// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNConvertNode : public MKLDNNNode {
public:
    MKLDNNConvertNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNConvertNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    void setDescs(const InferenceEngine::TensorDesc& input, const InferenceEngine::TensorDesc& output) {
        this->input = input;
        this->output = output;
    }

    const InferenceEngine::TensorDesc& getInput() { return input; }
    const InferenceEngine::TensorDesc& getOutput() { return output; }
private:
    InferenceEngine::TensorDesc input;
    InferenceEngine::TensorDesc output;
};
}  // namespace MKLDNNPlugin

