// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNDeconvolutionNode : public MKLDNNNode {
public:
    MKLDNNDeconvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNDeconvolutionNode() override = default;

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

private:
    bool withBiases;
    bool withGroups;
    bool isDW;
    size_t groupNum = 1;
    std::vector<int> stride;
    std::vector<int> paddingL;
    std::vector<int> dilation;
    std::vector<int> paddingR;
    MKLDNNDims weightsDims;
    static Register<MKLDNNDeconvolutionNode> reg;
    InferenceEngine::Blob::Ptr biases;
    std::vector<std::shared_ptr<mkldnn::convolution_forward::desc>> descs_fwd;
    std::vector<std::shared_ptr<mkldnn::convolution_backward_data::desc>> descs_bwd;
};

}  // namespace MKLDNNPlugin

