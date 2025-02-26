// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common/dnnl_executor.h"
#include "node.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Convolution : public Node {
public:
    Convolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override{};
    void selectOptimalPrimitiveDescriptor() override;
    void initSupportedPrimitiveDescriptors() override;
    int registerToAllocationContext(int offset, AllocationContext& context) override;
    void createPrimitive() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    ov::element::Type getRuntimePrecision() const override;

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
        return getInputShapeAtPort(WEIGHTS).getDims();
    }
    const std::vector<size_t>& getStride() {
        return m_attrs.stride;
    }
    const std::vector<size_t>& getDilation() {
        return m_attrs.dilation;
    }
    const std::vector<ptrdiff_t>& getPaddingL() {
        return m_attrs.paddingL;
    }
    const std::vector<ptrdiff_t>& getPaddingR() {
        return m_attrs.paddingR;
    }

    bool canFuse(const NodePtr& node) const override;
    bool isDepthWise() const {
        return m_attrs.isGrouped && 1 == groupOC && 1 == groupIC;
    }

protected:
    void addFusedNode(const NodePtr& fusingNode) override;
    void redefineOutputMemory(const std::vector<VectorDims>& newOutputShapes) override;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

private:
    class FusedSubgraph;
    using FusedSubgraphPtr = std::shared_ptr<FusedSubgraph>;
    using executorPtr = std::shared_ptr<DnnlExecutor>;

    std::tuple<ov::element::Type, ov::element::Type> getDstAndSumPrecision();
    std::tuple<VecMemoryDescs, MemoryDescPtr> initMemoryDescriptors(ov::element::Type dstType) const;
    ExecutorFactoryPtr<ConvAttrs> createExecutorFactory(const MemoryDescArgs& descs,
                                                        const ConvAttrs& attrs,
                                                        const PostOps& postOps);
    ExecutorPtr createFallbackExecutor();

    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void filterSupportedDescriptors();

    VectorDims makeInputDummyShape(const Shape& inpShape) const;

    enum InputId : uint8_t {
        DATA = 0,
        WEIGHTS,
        BIAS,
        WEIGHT_SCALES,
        WEIGHT_ZERO_POINTS,
        INPUT_SCALES,
        INPUT_ZERO_POINTS,
        OUTPUT_SCALES,
        OUTPUT_ZERO_POINTS,
    };

    std::unordered_map<size_t, size_t> m_atoi;  // memory argument id to input id
    ConvAttrs m_attrs;
    PostOps m_postOps;
    MemoryArgs m_memory;
    ExecutorFactoryPtr<ConvAttrs> m_factory;
    ExecutorPtr m_executor = nullptr;
    ExecutorPtr fallbackExecutor = nullptr;

    bool withSum;
    bool withDWConv;
    bool withSumBroadcast = false;

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

    FusedSubgraphPtr subgraph;
    std::unordered_map<NodePtr, std::vector<NodePtr>> fusedConstNodes;

    bool useJitPlanar = false;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
