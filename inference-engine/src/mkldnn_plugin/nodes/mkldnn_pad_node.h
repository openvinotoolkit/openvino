// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNPadNode : public MKLDNNNode {
public:
    MKLDNNPadNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

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
        int nThreads = 0;
        size_t nDimsForWork = 0lu;
        size_t workAmount = 0lu;
        size_t lastDstDim = 1lu;
        size_t shift = 0lu;
        uint8_t sizeData = 1;
    } params;

    template<typename T>
    struct PadConstantEmitter {
        void operator()(MKLDNNPadNode* node) {
            node->padConstantCommon<T>();
        }
    };

    std::string errorPrefix;
    static const size_t DATA_ID = 0;
    static const size_t PADS_BEGIN_ID = 1;
    static const size_t PADS_END_ID = 2;
    static const size_t PAD_VALUE_ID = 3;

    bool isPadValueSpecified = false;
};

}  // namespace MKLDNNPlugin
