// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common/dnnl_executor.h"
#include "node.h"
#include "oneapi/dnnl/dnnl.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Eltwise;

class Convolution : public Node {
public:
    Convolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void initDescriptor(const NodeConfig& config) override;
    void selectOptimalPrimitiveDescriptor() override;
    void initSupportedPrimitiveDescriptors() override;
    int registerToAllocationContext(int offset, AllocationContext& context) override;
    void createPrimitive() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    ov::element::Type getRuntimePrecision() const override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const override;

    dnnl::memory getWeights() const;
    dnnl::memory getBias() const;

    size_t descInputNumbers() override {
        return getOriginalInputsNumber();
    }

    bool canBeExecutedInInt8() const override;
    size_t getGroupNum() const {
        return groupNum;
    }
    // OV Legacy input zero point mechanism can support per-channel zero point.
    // Hold legacy input zero point.
    std::vector<uint8_t> legacyInputZeroPoints;
    // Hold legacy weight zero point.
    std::vector<float> legacyWeightsZeroPoints;
    // Hold legacy pre-calculated output compensation
    std::vector<int32_t> legacyOutputCompensation;
    // Hold stock per-tensor input zero point. Pass to onednn to calculate output compensation.
    std::vector<int32_t> inputZeroPoints;
    void initializeInputZeroPoints(const uint8_t* inputZpData, const size_t inputZpSize);

    const VectorDims& getWeightDims() {
        return weightDims;
    }
    const std::vector<size_t>& getStride() {
        return stride;
    }
    const std::vector<ptrdiff_t>& getDilation() {
        return dilation;
    }
    const std::vector<ptrdiff_t>& getPaddingL() {
        return paddingL;
    }
    const std::vector<ptrdiff_t>& getPaddingR() {
        return paddingR;
    }

    bool canFuse(const NodePtr& node) const override;
    bool isDepthWise() const {
        return isGrouped && 1 == groupOC && 1 == groupIC;
    }

protected:
    ov::element::Type fusedEltwisePrecision(const NodePtr& fusingNode) const;
    void redefineOutputMemory(const std::vector<VectorDims>& newOutputShapes) override;
    void addFusedNode(const NodePtr& fusingNode) override;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

private:
    enum class zpType { None, PerTensor, PerChannel };

    class FusedSubgraph;
    using FusedSubgraphPtr = std::shared_ptr<FusedSubgraph>;
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;

    class ConvolutionExecutor : public DnnlExecutor {
    public:
        ConvolutionExecutor(const dnnl::primitive_desc& pd,
                            const dnnl::memory::desc& inMemDesc,
                            const dnnl::memory::desc& weightMemDesc,
                            const dnnl::memory::desc& outMemDesc,
                            const dnnl::engine& engine,
                            bool constWeight);
    };

    class ConvolutionSumExecutor : public DnnlExecutor {
    public:
        ConvolutionSumExecutor(const dnnl::primitive_desc& pd,
                               const dnnl::memory::desc& inMemDesc,
                               const dnnl::memory::desc& weightMemDesc,
                               const dnnl::memory::desc& outMemDesc,
                               const dnnl::engine& engine,
                               bool constWeight);

    private:
        void reorder_exec(std::unordered_map<int, dnnl::memory> primArgs, const dnnl::stream& strm) override;
    };

    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void addLegacyZeroPoints(dnnl::primitive_attr& attr);
    void addZeroPoints(dnnl::primitive_attr& attr);
    void setPostOps(dnnl::primitive_attr& attr,
                    const VectorDims& dims,
                    bool useLegacyPostOps,
                    bool initWeights = false);
    void SetPostOpsAndZeroPoints(std::vector<dnnl::primitive_attr>& attrs);
    void filterSupportedDescriptors();
    bool isNspcAvailable() const;

    void updatePadding();
    MemoryDescPtr getSumMemDesc(const dnnl::primitive_desc& primitive_desc_it);
    MemoryPtr getOutputMemory() const;
    VectorDims makeInputDummyShape(const Shape& inpShape) const;
    VectorDims outputStaticShape() const;
    void appendLegacyZeroPointsArgs();
    void appendZeroPointsArgs();

    bool withBiases;
    bool withSum;
    bool withDWConv;
    bool isGrouped;
    bool withSumBroadcast = false;
    bool preferLegacyPostOps = false;
    bool preferLegacyZeroPoint = false;
    zpType inputZeroPointType = zpType::None;
    // maps each supportedPrimitiveDescriptor to corresponding desc from descs
    std::vector<size_t> descIdx;
    VectorDims expectedBiasDims{};

    std::vector<size_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    VectorDims weightDims;
    std::unordered_map<int, MemoryPtr> convPostOpsArgs[2];

    size_t dw_conv_oc;
    size_t dw_conv_ih;
    size_t dw_conv_iw;
    std::vector<size_t> dw_conv_kernel;
    std::vector<size_t> dw_conv_strides;
    dnnl::memory::data_type dw_conv_in_dt;

    size_t groupNum;
    size_t IC;
    size_t groupIC;
    size_t groupOC;

    ov::element::Type eltwisePrecision;

    const size_t X_AXIS = 0;
    const size_t Y_AXIS = 1;

    const bool isBrgConvAvailable();
    std::vector<dnnl::primitive_attr> attrs;
    AttrPtr pAttr;
    bool autoPadding = false;
    FusedSubgraphPtr subgraph;
    std::unordered_map<NodePtr, std::vector<NodePtr>> fusedConstNodes;

    MemoryPtr legacyInputZeroPointsMemPtr;
    MemoryPtr legacyWeightsZeroPointsMemPtr;
    MemoryPtr legacyOutputCompensationMemPtr;
    MemoryPtr stockInputZeroPointsMemPtr;
    dnnl::memory::data_type outputDataType = dnnl::memory::data_type::undef;
    ov::element::Type sumPrc = ov::element::dynamic;
    bool useJitPlanar = false;
    // TODO: migrate on convolution_auto algorithm for x64
#if defined(OPENVINO_ARCH_X86_64)
    const dnnl::algorithm baseConvAlgorithm = dnnl::algorithm::convolution_direct;
#else
    const dnnl::algorithm baseConvAlgorithm = dnnl::algorithm::convolution_auto;
#endif
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
