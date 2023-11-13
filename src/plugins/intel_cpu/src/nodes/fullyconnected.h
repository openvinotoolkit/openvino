// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>
#include "common/dnnl_executor.h"

namespace ov {
namespace intel_cpu {
namespace node {

class FullyConnected : public Node {
public:
    FullyConnected(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape &dims) const override;
    void getSupportedDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() == 3 ? 2 : 1;
    }

    const std::vector<impl_desc_type>& getDefaultImplPriority() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;

    size_t descInputNumbers() override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const override;

    ov::element::Type getRuntimePrecision() const override;

    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeExecutedInInt8() const override;
    void keepWeightsNonTransposed(bool weightsNonTransposed) {
        this->weightsNonTransposed = weightsNonTransposed;
    }

    void fuseDecompressionMultiply(const MemoryCPtr& memory);
    void fuseDecompressionSubtract(const MemoryCPtr& memory);

private:
    void createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                  const dnnl::memory::desc &outputDesc);

    VectorDims makeDummyInputDims() const;
    VectorDims makeDummyOutputDims(const VectorDims& inDims) const;

    VectorDims inDims;
    VectorDims outDims;

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &dims);

    bool withBiases = false;

    std::string errorPrefix;
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;
    dnnl::memory::data_type outputDataType = dnnl::memory::data_type::undef;

    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;
    bool useConv1x1 = false;
    impl_desc_type implementationTypeIP = impl_desc_type::unknown;
    MemoryDescPtr weightDescIP;
    dnnl::primitive_attr attr;

    static dnnl::convolution_forward::primitive_desc
    createDescriptorInternalForConv(DnnlMemoryDescCPtr inputDescPtr,
                                    DnnlMemoryDescCPtr weightDescPtr,
                                    DnnlMemoryDescCPtr biasDescPtr,
                                    DnnlMemoryDescCPtr outputDescPtr,
                                    const dnnl::primitive_attr& attr,
                                    const dnnl::engine& engine);

    bool canBeExecutedInConv1x1() const;
    void fuseDecompressionConstant(const MemoryCPtr& memory, MemoryCPtr& decompressionValuesPtr);

    // sparse weights
    bool useSparseWeights = false;
    float minSparseRate = 1.f;
    float weiSparseRate = 0.f;
    bool useSparseWeightsDecompression();
    VectorDims expectedBiasDims {};
    bool useMlas = false;
#ifdef OV_CPU_WITH_MLAS
    int64_t M, N, K;
    MemoryPtr mlasPackedPtr = nullptr;
    void executeMLAS();
    void prepackMLASWeight();
#endif
#if defined(OV_CPU_WITH_ACL)
    void prepareWeightsUsingDummyShape();
#endif
    bool useWeightsDecompressionImpl = false;
    MemoryCPtr decompressionSubtractPtr = nullptr;
    MemoryCPtr decompressionMultiplyPtr = nullptr;

    // FC with transposed weights
    bool weightsNonTransposed = false;
    DnnlMemoryDescPtr makeTransposedWeightDescriptor(DnnlMemoryDescPtr desc);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
