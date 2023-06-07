// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_matmul.hpp"
#include "ie_parallel.hpp"
#include <dnnl_extension_utils.h>
#include "nodes/executors/matmul.hpp"
#include "onednn/dnnl.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

DnnlMatMulExecutor::DnnlMatMulExecutor(const ExecutorContext::CPtr context) : MatMulExecutor(context) {}

bool DnnlMatMulExecutor::init(const MatMulAttrs& matmulAttrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs,
                              const dnnl::primitive_attr &attr) {
    auto engine = context->getEngine();
    this->stream = dnnl::stream(engine);
    this->matmulAttrs = matmulAttrs;
    auto localAttrs = dnnl::primitive_attr(attr.get()->clone());
    localAttrs.set_scratchpad_mode(dnnl::scratchpad_mode::user);


    auto getPrimitiveDesc = [&]() {
        auto prim_desc = createDescriptor(engine, matmulAttrs, srcDescs, dstDescs, attr);
        auto first_desc = dnnl::matmul::primitive_desc(prim_desc.get());

        if (!context->getImplPriorities().empty()) {
            for (auto preferredImplType : context->getImplPriorities()) {
                const bool found = DnnlExtensionUtils::find_implementation(prim_desc, preferredImplType);

                if (found)
                    return prim_desc;
            }
        }

        return first_desc;
    };

    auto prim_desc = getPrimitiveDesc();
    implType = parse_impl_name(prim_desc.impl_info_str());

    if (!prim_desc)
        return false;

    auto scratchpadMemoryDesc = DnnlExtensionUtils::makeDescriptor(prim_desc.query_md(dnnl::query::scratchpad_md));
    scratchpadMemory = context->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);

    prim = std::make_shared<dnnl::matmul>(prim_desc);

    return true;
}

void DnnlMatMulExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    std::unordered_map<int, dnnl::memory> primArgs;


    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMemory->GetPrimitive();
    primArgs[DNNL_ARG_SRC_0] = src[0]->GetPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src[1]->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dst[0]->GetPrimitive();
    if (matmulAttrs.withBias)
        primArgs[DNNL_ARG_BIAS] = src[2]->GetPrimitive();

    for (auto & entry : postOpsArgs) {
        primArgs[entry.first] = entry.second->GetPrimitive();
    }

    (*prim).execute(stream, primArgs);
}

std::pair<Shape, Shape> DnnlMatMulExecutor::makeDummyInputShapes(const MatMulAttrs& matmulAttrs, const Shape& in0, const Shape& in1) {
    if (in0.getRank() < 2 || in1.getRank() < 2) {
        IE_THROW() << "Can't create dummy inputs with rank less 2";
    }

    if (in0.getRank() != in1.getRank()) {
        IE_THROW() << "Can't create dummy inputs if input's rank not equal";
    }

    auto swapTranspDims = [&](VectorDims& in0, VectorDims& in1) {
        if (matmulAttrs.transposeA) {
            std::swap(in0[in0.size() - 1], in0[in0.size() - 2]);
        }
        if (matmulAttrs.transposeB) {
            std::swap(in1[in1.size() - 1], in1[in1.size() - 2]);
        }
    };

    auto inDims0 = in0.getDims();
    auto inDims1 = in1.getDims();

    auto minDims0 = in0.getMinDims();
    auto maxDims0 = in0.getMaxDims();
    auto minDims1 = in1.getMinDims();
    auto maxDims1 = in1.getMaxDims();

    swapTranspDims(inDims0, inDims1);
    swapTranspDims(minDims0, minDims1);
    swapTranspDims(maxDims0, maxDims1);

    auto fillDummy = [&](size_t idx0, size_t idx1) {
        if (inDims0[idx0] == Shape::UNDEFINED_DIM && inDims1[idx1] == Shape::UNDEFINED_DIM) {
            inDims0[idx0] = inDims1[idx1] = std::min(std::min(maxDims0[idx0], maxDims1[idx1]),
                                            std::max(std::max(minDims0[idx0], minDims1[idx1]), static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
        } else {
            if (inDims0[idx0] == Shape::UNDEFINED_DIM && inDims1[idx1] != Shape::UNDEFINED_DIM) {
                if (inDims1[idx1] == 1 && minDims0[idx0] != Shape::UNDEFINED_DIM) {
                    inDims0[idx0] = std::max<Dim>(minDims0[idx0], 1);
                } else {
                    inDims0[idx0] = inDims1[idx1];
                }
            } else if (inDims0[idx0] != Shape::UNDEFINED_DIM && inDims1[idx1] == Shape::UNDEFINED_DIM) {
                if (inDims0[idx0] == 1 && minDims1[idx1] != Shape::UNDEFINED_DIM) {
                    inDims1[idx1] = std::max<Dim>(minDims1[idx1], 1);
                } else {
                    inDims1[idx1] = inDims0[idx0];
                }
            }
        }
    };

    // fill k
    fillDummy(inDims0.size() - 1, inDims1.size() - 2);

    // fill m, n
    if (inDims0[inDims0.size() - 2] == Shape::UNDEFINED_DIM) {
        inDims0[inDims0.size() - 2] = std::min(maxDims0[inDims0.size() - 2],
                                               std::max(minDims0[inDims0.size() - 2], static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
    }
    if (inDims1[inDims1.size() - 1] == Shape::UNDEFINED_DIM) {
        inDims1[inDims1.size() - 1] = std::min(maxDims1[inDims1.size() - 1],
                                               std::max(minDims1[inDims1.size() - 1], static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
    }

    // fill batches
    for (size_t i = 0; i < inDims0.size() - 2; i++) {
        fillDummy(i, i);
    }

    swapTranspDims(inDims0, inDims1);

    return {Shape(inDims0), Shape(inDims1)};
}

DnnlMatMulExecutor::Key::Key(const MatMulAttrs& matmulAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) :
    matmulAttrs(matmulAttrs),
    inp0(MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[0])),
    inp1(MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[1])),
    bias(matmulAttrs.withBias ? MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[2]) : nullptr),
    out(MemoryDescUtils::convertToDnnlMemoryDesc(dstDescs[0])),
    attr(attr) {}

size_t DnnlMatMulExecutor::Key::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, matmulAttrs.transposeA);
    seed = hash_combine(seed, matmulAttrs.transposeB);
    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool DnnlMatMulExecutor::Key::operator==(const Key& rhs) const {
    bool retVal = true;
    retVal = retVal && matmulAttrs.transposeA == rhs.matmulAttrs.transposeA;
    retVal = retVal && matmulAttrs.transposeB == rhs.matmulAttrs.transposeB;

    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get();
    return retVal;
}

}   // namespace intel_cpu
}   // namespace ov
