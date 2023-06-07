// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/matmul.hpp"
#include "dnnl.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {

class DnnlMatMulExecutor : public MatMulExecutor {
public:
    DnnlMatMulExecutor(const ExecutorContext::CPtr context);

    bool init(const MatMulAttrs& matmulAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

    static dnnl::matmul::primitive_desc createDescriptor(const dnnl::engine& engine,
                                                         const MatMulAttrs& matmulAttrs,
                                                         const std::vector<MemoryDescPtr>& srcDescs,
                                                         const std::vector<MemoryDescPtr>& dstDescs,
                                                         const dnnl::primitive_attr &attr) {
        auto inputShape0 = srcDescs[0]->getShape();
        const VectorDims inStrides0 = getStridesAndModifyShape(inputShape0, matmulAttrs.transposeA);
        auto inDataDesc0 = std::make_shared<DnnlBlockedMemoryDesc>(srcDescs[0]->getPrecision(), inputShape0, inStrides0);

        auto inputShape1 = srcDescs[1]->getShape();
        const VectorDims inStrides1 = getStridesAndModifyShape(inputShape1, matmulAttrs.transposeB);
        auto inDataDesc1 = std::make_shared<DnnlBlockedMemoryDesc>(srcDescs[1]->getPrecision(), inputShape1, inStrides1);

        auto outputShape = dstDescs[0]->getShape();
        auto outDataDesc = std::make_shared<DnnlBlockedMemoryDesc>(dstDescs[0]->getPrecision(), outputShape);

        dnnl::matmul::primitive_desc matmul_desc;
        if (matmulAttrs.withBias) {
            // oneDNN matmul requires shape for bias desc to be the same rank
            VectorDims biasDims(outputShape.getRank(), 1);
            const auto& outDims = outputShape.getStaticDims();
            const auto chIdx = outputShape.getRank() - 1;
            biasDims[chIdx] = outDims[chIdx];
            const auto bdt = DnnlExtensionUtils::IEPrecisionToDataType(srcDescs[2]->getPrecision());
            auto biasDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(biasDims), bdt, dnnl::memory::format_tag::any);

            matmul_desc = dnnl::matmul::primitive_desc(engine,
                                                       inDataDesc0->getDnnlDesc(),
                                                       inDataDesc1->getDnnlDesc(),
                                                       biasDesc,
                                                       outDataDesc->getDnnlDesc(),
                                                       attr);
        } else {
            matmul_desc = dnnl::matmul::primitive_desc(engine,
                                                       inDataDesc0->getDnnlDesc(),
                                                       inDataDesc1->getDnnlDesc(),
                                                       outDataDesc->getDnnlDesc(),
                                                       attr);
        }

        return matmul_desc;
    }

    /* Example MatMul:
    * 2x128x512(T) * 2x128x512 = 2x512x512
    * First input 2x128x512(T) should be transposed
    * oneDNN requires memory::desc for this input to:
    * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
    * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
    */
    static VectorDims getStridesAndModifyShape(Shape& shape, const bool transpose) {
        const auto getRank = shape.getRank();

        VectorDims strides(getRank, 1);
        const auto& staticDims = shape.getStaticDims();
        for (size_t i = 1; i < getRank; i++) {
            strides[getRank - i - 1 ] = strides[getRank - i] * staticDims[getRank - i];
        }

        if (transpose && getRank > 1) {
            // form new shape
            auto dims = staticDims;
            std::swap(dims[getRank - 2], dims[getRank - 1]);
            shape = Shape{dims};
            // update strides
            strides[getRank - 1] = staticDims[getRank - 2];
            strides[getRank - 2] = 1;
        }

        return strides;
    }

    struct Key {
        const MatMulAttrs& matmulAttrs;
        DnnlMemoryDescPtr inp0;
        DnnlMemoryDescPtr inp1;
        DnnlMemoryDescPtr bias;
        DnnlMemoryDescPtr out;
        const dnnl::primitive_attr& attr;

        Key(const MatMulAttrs& matmulAttrs,
            const std::vector<MemoryDescPtr>& srcDescs,
            const std::vector<MemoryDescPtr>& dstDescs,
            const dnnl::primitive_attr &attr);
        size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

private:
    static std::pair<Shape, Shape> makeDummyInputShapes(const MatMulAttrs& matmulAttrs, const Shape& in0, const Shape& in1);

    dnnl::stream stream;

    MatMulAttrs matmulAttrs;
    std::shared_ptr<dnnl::matmul> prim;
    MemoryPtr scratchpadMemory;
    impl_desc_type implType = impl_desc_type::undef;
};

class DnnlMatMulExecutorBuilder : public MatMulExecutorBuilder {
public:
    bool isSupported(const MatMulAttrs& matmulAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs,
                     const dnnl::primitive_attr &attr) const override {
        // TODO: add correct conditions
        return true;
    }

    MatMulExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<DnnlMatMulExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov
