// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNPadNode : public MKLDNNNode {
public:
    MKLDNNPadNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNPadNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    enum PadMode {
        CONSTANT = 0,
        EDGE = 1,
        REFLECT = 2,
        SYMMETRIC = 3
    };

    void padConstantOrEdge(const float *srcData, float* dstData,
                           InferenceEngine::SizeVector srcDims, InferenceEngine::SizeVector dstDims,
                           const bool isEdge = false);
    void padReflectOrSymmetric(const float *srcData, float* dstData,
                               InferenceEngine::SizeVector srcDims, InferenceEngine::SizeVector dstDims,
                               const bool isSymmetric = false);

    size_t getWorkAmountDst() const;

    PadMode padMode = CONSTANT;
    float padValue = 0.f;
    std::vector<unsigned int> padsBegin;
    std::vector<unsigned int> padsEnd;

    struct {
        InferenceEngine::SizeVector srcODims;
        InferenceEngine::SizeVector srcStrides;
        InferenceEngine::SizeVector dstStrides;
        size_t padPointsNum = 0;
        InferenceEngine::SizeVector padDims;
    } params;
};

}  // namespace MKLDNNPlugin
