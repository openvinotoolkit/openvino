// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class FullyConnected : public Node {
public:
    FullyConnected(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape &dims) const override;
    void getSupportedDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    size_t getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() == 3 ? 2 : 1;
    }

    const std::vector<impl_desc_type>& getPrimitivesPriority() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;

    size_t descInputNumbers(DnnlDesriptor desc) override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    void initSupportedPrimitiveDescriptors() override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    std::shared_ptr<dnnl::primitive_attr> initPrimitiveAttr() override;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

    void setDynamicBatchLim(int lim) override;

private:
    void createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                  const dnnl::memory::desc &outputDesc);

    VectorDims makeDummyInputDims() const;
    VectorDims makeDummyOutputDims(const VectorDims& inDims) const;

    VectorDims inDims;
    VectorDims outDims;

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims, bool initWeights = false);

    bool withBiases = false;

    std::string errorPrefix;
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
