// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <ie_common.h>
#include <string>
#include <vector>
#include <array>

namespace MKLDNNPlugin {

class MKLDNNMatMulNode : public MKLDNNNode {
public:
    MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void initSupportedPrimitiveDescriptors() override;
    MemoryDescPtr getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    void createPrimitive() override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    bool created() const override;
    size_t getMaxBatch() const override;

    InferenceEngine::Precision getRuntimePrecision() const override;
    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return getOriginalInputsNumber();
    }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr() const override;

private:
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights) const;

    std::string errorPrefix;

    /* whether to transpose input */
    std::array<bool, 2> transposeIn;
    /* initial shapes without transpose,
     * necessary to hide transpose effect from plugin */
    std::array<Shape, 2> initialInShapes;

    std::array<MemoryDescPtr, 2> inDataDesc;
    MemoryDescPtr outDataDesc;
};

}  // namespace MKLDNNPlugin

