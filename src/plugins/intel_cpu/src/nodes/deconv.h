// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "common/dnnl_executor.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/deconv.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Deconvolution : public Node {
public:
    Deconvolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    size_t descInputNumbers() override {
        return static_cast<size_t>(getParentEdges().size());
    }

    std::shared_ptr<MemoryDesc> getSrcMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const override;

    ov::element::Type getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    bool canFuse(const NodePtr& node) const override;

    const VectorDims& getWeightDims() const {
        return getInputShapeAtPort(1).getStaticDims();
    }
    const std::vector<ptrdiff_t>& getStride() const {
        return deconvAttrs.stride;
    }

    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    bool needShapeInfer() const override;

    bool canFuseBias() const;
    bool canBeExecutedInInt8() const override;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

protected:
    AttrPtr initPrimitiveAttr() override;
    AttrPtr makePrimitiveAttr(const VectorDims& dims);
    std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape& dims) const override;
    std::shared_ptr<DeconvExecutor> execPtrDeconvACL = nullptr;

private:
    using executorPtr = std::shared_ptr<DnnlExecutorLegacy>;
    executorPtr execPtr = nullptr;
    class DeconvDNNLExecutor : public DnnlExecutorLegacy {
    public:
        DeconvDNNLExecutor(const dnnl::deconvolution_forward::primitive_desc& pd,
                           const dnnl::memory::desc& inMemDesc,
                           const dnnl::memory::desc& weightMemDesc,
                           const dnnl::memory::desc& outMemDesc,
                           const dnnl::engine& engine,
                           bool constWeight);
    };

    bool isImplicit1x1PaddingAsymmetric(const VectorDims& inputDims);
    bool withGroups = false;
    bool isDW = false;
    bool isInt8 = false;
    bool autoPad = false;
    bool externOutShape = false;
    size_t groupNum = 1;
    size_t IC = 0;
    size_t OC = 0;
    std::vector<int32_t> lastOutputSpatialDims;
    VectorDims dnnlCompatibleWeiDims;
    VectorDims expectedBiasDims;

    bool useACL = false;
    DeconvAttrs deconvAttrs;

    Shape inShape, outShape;

    AttrPtr pAttr;

    dnnl::memory::data_type outputDataType = dnnl::memory::data_type::undef;
    MemoryPtr dnnlCompatibleWeights = nullptr;

    std::shared_ptr<dnnl::primitive_attr> attr;
    void setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims);
    VectorDims shapeInferInternal(const VectorDims& inDims, std::vector<int32_t> outSpDims) const;
    void initPaddingR(const Shape& inShape, const Shape& outShape);
    std::vector<int32_t> readOutputSpatialDims() const;
    std::pair<VectorDims, VectorDims> makeDummyInOutShape();
    bool withBiases = false;
    size_t biasPort;

    void createDnnlCompatibleWeights();
    bool weightIsConst = false;
    bool asymmetricPaddingAnd1x1 = false;
    bool is1x1 = false;
    bool isConstOutShape = false;
};

}  // namespace ov::intel_cpu::node
