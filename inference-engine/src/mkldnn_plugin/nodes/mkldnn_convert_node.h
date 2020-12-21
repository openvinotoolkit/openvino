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
        this->input.reset(new InferenceEngine::TensorDesc(input));
        this->output.reset(new InferenceEngine::TensorDesc(output));
    }

    std::shared_ptr<const InferenceEngine::TensorDesc> getInput() const { return input; }
    std::shared_ptr<const InferenceEngine::TensorDesc> getOutput() const { return output; }
private:
    std::shared_ptr<InferenceEngine::TensorDesc> input;
    std::shared_ptr<InferenceEngine::TensorDesc> output;
};
}  // namespace MKLDNNPlugin

