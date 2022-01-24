// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <ie_common.h>
#include <string>
#include <vector>
#include <array>
#include "memory_desc/dnnl_blocked_memory_desc.h"

namespace MKLDNNPlugin {

class MKLDNNMatMulNode : public MKLDNNNode {
public:
    MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void initSupportedPrimitiveDescriptors() override;
    MemoryDescPtr getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    bool created() const override;
    size_t getMaxBatch() const override;

    InferenceEngine::Precision getRuntimePrecision() const override;
    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return getOriginalInputsNumber();
    }

    size_t getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() - 1;
    }

    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    const std::vector<impl_desc_type>& getPrimitivesPriority() override;

protected:
    AttrPtr initPrimitiveAttr() override;
    AttrPtr initPrimitiveAttr(const VectorDims& dims);

private:
    mkldnn::memory::desc getBiasDescFrom(const DnnlMemoryDescCPtr outMemDesc);

    bool withBiases;

    void setPostOps(mkldnn::primitive_attr &attr, const VectorDims& dims, bool initWeights);

    std::string errorPrefix;

    /* whether to transpose input */
    std::array<bool, 2> transposeIn;

    std::array<DnnlBlockedMemoryDescPtr, 2> inDataDesc;
    DnnlBlockedMemoryDescPtr outDataDesc;
};

}  // namespace MKLDNNPlugin

