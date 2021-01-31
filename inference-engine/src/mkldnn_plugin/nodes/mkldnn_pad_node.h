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

    void padConstant();
    template<typename T> void padConstantCommon();
    void padConstantZero();
    void padEdge();
    void padReflectOrSymmetric(const bool isSymmetric = false);

    inline void getDstIdx(const InferenceEngine::SizeVector& indexes, size_t& dstIdx) const;

    PadMode padMode = CONSTANT;
    float padValue = 0.f;
    std::vector<unsigned int> padsBegin;
    std::vector<unsigned int> padsEnd;

    struct {
        InferenceEngine::SizeVector srcDims;
        InferenceEngine::SizeVector dstDims;
        InferenceEngine::SizeVector srcODims;
        InferenceEngine::SizeVector srcStrides;
        InferenceEngine::SizeVector dstStrides;
        InferenceEngine::SizeVector srcDimsForReflectOrSymmetric;
        size_t nDimsForWork;
        size_t workAmount;
        size_t lastDstDim;
        size_t shift;
        uint8_t sizeData;
    } params;

    template<typename T>
    struct PadConstantEmitter {
        void operator()(MKLDNNPadNode* node) {
            node->padConstantCommon<T>();
        }
    };
};

}  // namespace MKLDNNPlugin
