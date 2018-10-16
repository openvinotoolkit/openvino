// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNBatchNormalizationNode : public MKLDNNNode {
public:
    MKLDNNBatchNormalizationNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);

    ~MKLDNNBatchNormalizationNode() override = default;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void createPrimitive() override;
    bool created() const override;
    bool fusedWithScale() const {return fusedWith.size() == 1 && fusedWith[0]->getType() == Depthwise
                                        && fusedWith[0]->getCnnLayer()->type == "ScaleShift";}

    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    MKLDNNMemoryDesc getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
private:
    static Register<MKLDNNBatchNormalizationNode> reg;
    float eps = 0.0f;
    MKLDNNMemoryDesc GetVarianceDesc(const mkldnn::memory::primitive_desc& primitive_desc) const;
    MKLDNNMemoryDesc GetMeanDesc(const mkldnn::memory::primitive_desc& primitive_desc) const;
    MKLDNNMemoryDesc GetScaleShiftWeightsDesc(const mkldnn::memory::primitive_desc& primitive_desc) const;
};

}  // namespace MKLDNNPlugin

