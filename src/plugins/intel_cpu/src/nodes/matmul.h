// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/dnnl_executor.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "node.h"

#include <array>

namespace ov {
namespace intel_cpu {
namespace node {

class MatMul : public Node {
public:
    MatMul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void initSupportedPrimitiveDescriptors() override;
    MemoryDescPtr getSrcMemDesc(const dnnl::primitive_desc &prim_desc, size_t idx) const override;
    bool canFuse(const NodePtr& node) const override;
    bool created() const override;

    ov::element::Type getRuntimePrecision() const override;
    size_t descInputNumbers() override {
        return getOriginalInputsNumber();
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() - 1;
    }

    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;
    bool canBeExecutedInInt8() const override;

    bool isExecutable() const override;

protected:
    AttrPtr initPrimitiveAttr() override;
    AttrPtr initPrimitiveAttr(const VectorDims& dims);

private:
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;
    dnnl::memory::desc getBiasDescFrom(const DnnlMemoryDescCPtr outMemDesc);
    std::pair<Shape, Shape>
    makeDummyInputShapes(const Shape& in0, const Shape& in1, const Shape& out) const;

    bool withBiases;

    void setPostOps(dnnl::primitive_attr &attr, const VectorDims& dims, bool initWeights);

    std::string errorPrefix;

    /* whether to transpose input */
    std::array<bool, 2> transposeIn;

    std::array<DnnlBlockedMemoryDescPtr, 2> inDataDesc;
    DnnlBlockedMemoryDescPtr outDataDesc;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
