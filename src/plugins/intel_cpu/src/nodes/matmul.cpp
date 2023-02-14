// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.h"

#include "ie_precision.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "eltwise.h"

#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "fake_quantize.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <dnnl_extension_utils.h>
#include <common/primitive_hashing_utils.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct MatMulKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    dnnl::primitive_attr attr;
    impl_desc_type implType;

    size_t hash() const;
    bool operator==(const MatMulKey& rhs) const;
};

size_t MatMulKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(ptr->getDnnlDesc().data));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    seed = hash_combine(seed, implType);
    return seed;
}

bool MatMulKey::operator==(const MatMulKey &rhs) const {
    bool retVal = true;
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
    retVal = retVal && *attr.get() == *rhs.attr.get() &&
             implType == rhs.implType;
    return retVal;
}

bool canBeExecutedInInt8(const Precision& firstInput, const Precision& secondInput) {
    return one_of(firstInput, Precision::U8, Precision::I8) && secondInput == Precision::I8;
}
} // namespace

bool MatMul::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MatMul::MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
    Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)), withBiases(false) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    if (!matMul) {
        IE_THROW(NotImplemented) << "Operation with name " << op->get_friendly_name() << ":" << op->get_type_name() <<
            " is not an instance of MatMul from opset1";
    }

    transposeIn[0] = matMul->get_transpose_a();
    transposeIn[1] = matMul->get_transpose_b();
}

bool MatMul::canFuse(const NodePtr& node) const {
    // per channel binary post op for rank > 2D is supported only by oneDNN reference implementation because of unusual MatMul channel axis (issue 6669)
    if (getOutputShapeAtPort(0).getRank() > 2) {
        if (const auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get())) {
            if (one_of(eltwiseNode->getAlgorithm(), Algorithm::EltwiseAdd,
                                                    Algorithm::EltwiseMultiply,
                                                    Algorithm::EltwiseSubtract,
                                                    Algorithm::EltwiseDivide,
                                                    Algorithm::EltwisePrelu,
                                                    Algorithm::EltwiseMulAdd,
                                                    Algorithm::EltwisePowerStatic) &&
                eltwiseNode->getBroadcastingPolicy() != Eltwise::PerTensor) {
                return false;
            }
        } else if (const auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get())) {
            if (fakeQuantizeNode->getBroadcastingPolicy() != FakeQuantize::PerTensor) {
                return false;
            }
        }
    }

    // Todo:
    //  Consider the case when Matmul doesn't support execution in int8, but is getting fused with FQ with int8 output.
    //  Then the Matmul will change its output precision to fp32, but the FQ child will still has the int8 input precision.
    //  This information should be propagated! Note that we may need to propagate updated precision to child fused nodes.
    if (node->getType() == Type::FakeQuantize &&
        one_of(node->getOriginalOutputPrecisionAtPort(0), Precision::I8, Precision::U8) &&
        !canBeExecutedInInt8(getOriginalInputPrecisionAtPort(0), getOriginalInputPrecisionAtPort(1)))
        return false;
    return canFuseSimpleOperation(node);
}

void MatMul::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims, bool initWeights = false) {
    dnnl::post_ops ops;

    dnnl::memory::data_type outputDataType = dnnl::memory::data_type::undef;
    if (outDataDesc) {
        outputDataType = outDataDesc->getDataType();
    }

    bool isINT8 = canBeExecutedInInt8(getOriginalInputPrecisionAtPort(0), getOriginalInputPrecisionAtPort(1));

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8);

    for (int i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

Node::AttrPtr MatMul::initPrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims, true);

    (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    return attr;
}

Node::AttrPtr MatMul::initPrimitiveAttr() {
    auto dummyShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
    return initPrimitiveAttr(dummyShape.getStaticDims());
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

dnnl::memory::desc MatMul::getBiasDescFrom(const DnnlMemoryDescCPtr outMemDesc) {
    // oneDNN matmul requires shape for bias desc to be the same rank
    VectorDims biasDims(outMemDesc->getShape().getRank(), 1);
    const auto outDims = outMemDesc->getShape().getStaticDims();
    const auto chIdx = getFusingAxis();
    biasDims[chIdx] = outDims[chIdx];
    const auto bdt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(2));

    return dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(biasDims), bdt, memory::format_tag::any);
}

void MatMul::getSupportedDescriptors() {
    if (getParentEdges().size() != getOriginalInputsNumber())
        IE_THROW()  << errorPrefix << " has incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW()  << errorPrefix << " has incorrect number of output edges for layer " << getName();

    withBiases = getOriginalInputsNumber() == 3;

    auto firstInPortPrec = getOriginalInputPrecisionAtPort(0);
    auto secondInPortPrec = getOriginalInputPrecisionAtPort(1);
    auto outPortPrec = getOriginalOutputPrecisionAtPort(0);

    if (firstInPortPrec.size() != secondInPortPrec.size())
        firstInPortPrec = secondInPortPrec = getMaxPrecision(getOriginalInputPrecisions());

    // fallback to fp32 for any precision that cannot be handled natively
    if ((!one_of(firstInPortPrec , Precision::U8, Precision::I8, Precision::BF16, Precision::FP32) ||
         !one_of(secondInPortPrec , Precision::I8, Precision::BF16, Precision::FP32))) {
        outPortPrec = firstInPortPrec = secondInPortPrec = Precision::FP32;
    }

    if (!fusedWith.empty()) {
        outPortPrec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (!canBeExecutedInInt8(firstInPortPrec, secondInPortPrec) && one_of(outPortPrec, Precision::U8, Precision::I8))
        outPortPrec = Precision::FP32; // INT output is not supported for non-INT inputs

    const auto& inputShape0 = getInputShapeAtPort(0);
    const auto& inputShape1 = getInputShapeAtPort(1);
    const auto& outputShape = getOutputShapeAtPort(0);

    if (inputShape0.getRank() != inputShape1.getRank() || inputShape0.getRank() != outputShape.getRank())
        IE_THROW()  << errorPrefix << " has invalid dims count";

    const int nDims = inputShape0.getRank();
    const auto xAxis = nDims - 1;
    const auto yAxis = nDims - 2;
    const auto xAxis0 = transposeIn[0] ? yAxis : xAxis;
    const auto yAxis0 = transposeIn[0] ? xAxis : yAxis;
    const auto xAxis1 = transposeIn[1] ? yAxis : xAxis;
    const auto yAxis1 = transposeIn[1] ? xAxis : yAxis;

    const auto& inDims0 = getInputShapeAtPort(0).getDims();
    const auto& inDims1 = getInputShapeAtPort(1).getDims();
    const auto& outDims = getOutputShapeAtPort(0).getDims();

    // coverity[copy_paste_error]
    if (!dimsEqualWeak(inDims0[xAxis0], inDims1[yAxis1]) ||
        !dimsEqualWeak(inDims0[yAxis0], outDims[yAxis]) ||
        !dimsEqualWeak(inDims1[xAxis1], outDims[xAxis]))
        IE_THROW()  << errorPrefix << " has incorrect spatial input and output dimensions";

    for (int dim_idx = nDims - 3; dim_idx >= 0; dim_idx--) {
        if ((!dimsEqualWeak(inDims0[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims0[dim_idx], 1)) ||
            (!dimsEqualWeak(inDims1[dim_idx], outDims[dim_idx]) &&
             !dimsEqualWeak(inDims1[dim_idx], 1))) {
            IE_THROW()  << errorPrefix << " has incorrect input batch dimensions";
        }
    }

    std::vector<Shape> staticInputShapes{inputShape0, inputShape1};
    if (inputShape0.isDynamic() || inputShape1.isDynamic()) {
        std::tie(staticInputShapes[0], staticInputShapes[1]) = makeDummyInputShapes(inputShape0, inputShape1);
    }

    auto staticOutputShape = outputShape.isStatic() ? outputShape : Shape(shapeInferGeneric(staticInputShapes).front());

    const VectorDims inStrides0 = getStridesAndModifyShape(staticInputShapes[0], transposeIn[0]);
    const VectorDims inStrides1 = getStridesAndModifyShape(staticInputShapes[1], transposeIn[1]);

    inDataDesc[0] = std::make_shared<DnnlBlockedMemoryDesc>(firstInPortPrec, staticInputShapes[0], inStrides0);
    inDataDesc[1] = std::make_shared<DnnlBlockedMemoryDesc>(secondInPortPrec, staticInputShapes[1], inStrides1);
    outDataDesc   = std::make_shared<DnnlBlockedMemoryDesc>(outPortPrec, staticOutputShape);

    createDescriptor({inDataDesc[0], inDataDesc[1]}, {outDataDesc});
}

std::pair<Shape, Shape> MatMul::makeDummyInputShapes(const Shape& in0, const Shape& in1) const {
    if (in0.getRank() < 2 || in1.getRank() < 2) {
        IE_THROW() << "Can't create dummy inputs with rank less 2";
    }

    if (in0.getRank() != in1.getRank()) {
        IE_THROW() << "Can't create dummy inputs if input's rank not equal";
    }

    auto swapTranspDims = [&](VectorDims& in0, VectorDims& in1) {
        if (transposeIn[0]) {
            std::swap(in0[in0.size() - 1], in0[in0.size() - 2]);
        }
        if (transposeIn[1]) {
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

void MatMul::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                                        const std::vector<MemoryDescPtr>& outputDesc) {
    std::shared_ptr<dnnl::matmul::desc> matmul_desc;
    if (withBiases) {
        matmul_desc.reset(new matmul::desc(inDataDesc[0]->getDnnlDesc(),
                                           inDataDesc[1]->getDnnlDesc(),
                                           getBiasDescFrom(outDataDesc),
                                           outDataDesc->getDnnlDesc()));
    } else {
        matmul_desc.reset(new matmul::desc(inDataDesc[0]->getDnnlDesc(),
                                           inDataDesc[1]->getDnnlDesc(),
                                           outDataDesc->getDnnlDesc()));
    }

    descs.emplace_back(matmul_desc);
}

void MatMul::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto attr = initPrimitiveAttr();

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), *attr);
        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace(-1);
                portConfig.constant(false);
                portConfig.setMemDesc(getSrcMemDesc(itpd, i));

                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace(canBeInPlace() ? 0 : -1);
                portConfig.constant(false);
                portConfig.setMemDesc(getDstMemDesc(itpd, i));

                config.outConfs.push_back(portConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

MemoryDescPtr MatMul::getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1): primitive_desc_it.src_desc(idx);

    if (idx < 2) // inputs
        return std::make_shared<CpuBlockedMemoryDesc>(
            DnnlExtensionUtils::DataTypeToIEPrecision(static_cast<dnnl::memory::data_type>(desc.data.data_type)),
            getInputShapeAtPort(idx)); /* provide initial shapes, so hide transpose effect */
    else // bias
        return DnnlExtensionUtils::makeDescriptor(desc);
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

size_t MatMul::getMaxBatch() const {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MatMul::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void MatMul::prepareParams() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& src0MemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto& src1MemPtr = getParentEdgeAt(1)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW()  << errorPrefix << " did not allocate destination memory";
    if (!src0MemPtr || !src0MemPtr->isAllocated() || !src1MemPtr || !src1MemPtr->isAllocated())
        IE_THROW()  << errorPrefix << " did not allocate input memory";

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW()  << errorPrefix << " did not set preferable primitive descriptor";

    DnnlMemoryDescPtr src0TransposedDesc;
    DnnlMemoryDescPtr src1TransposedDesc;

    AttrPtr attr;

    if (isDynamicNode()) {
        attr = initPrimitiveAttr(dstMemPtr->getStaticDims());

        const auto& src0Desc = src0MemPtr->getDesc();
        const auto& src1Desc = src1MemPtr->getDesc();

        auto src0Shape = src0Desc.getShape();
        auto src0Strides = getStridesAndModifyShape(src0Shape, transposeIn[0]);
        src0TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src0Desc.getPrecision(), src0Shape, src0Strides);

        auto src1Shape = src1Desc.getShape();
        auto src1Strides = getStridesAndModifyShape(src1Shape, transposeIn[1]);
        src1TransposedDesc = std::make_shared<DnnlBlockedMemoryDesc>(src1Desc.getPrecision(), src1Shape, src1Strides);
    } else {
        attr = initPrimitiveAttr();
        src0TransposedDesc = inDataDesc[0];
        src1TransposedDesc = inDataDesc[1];
    }

    auto dstDnnlDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    DnnlMemoryDescPtr dnnlBiasMemDesc = nullptr;
    if (withBiases) {
        auto& biasMemory = getParentEdgeAt(2)->getMemoryPtr();
        if (!biasMemory || !biasMemory->isAllocated())
            IE_THROW()  << errorPrefix << " did not allocate bias memory";
        dnnlBiasMemDesc = biasMemory->GetDescWithType<DnnlMemoryDesc>();
    }

    MatMulKey key = {src0TransposedDesc, src1TransposedDesc, dnnlBiasMemDesc,
                     dstDnnlDesc, *attr, selected_pd->getImplementationType()};

    auto engine = getEngine();

    auto builder = [&engine](const MatMulKey& key) -> dnnl::primitive {
        std::shared_ptr<dnnl::matmul::desc> matmul_desc;

        if (key.bias) {
            matmul_desc.reset(new dnnl::matmul::desc{key.inp0->getDnnlDesc(),
                                                       key.inp1->getDnnlDesc(),
                                                       key.bias->getDnnlDesc(),
                                                       key.out->getDnnlDesc()});
        } else {
            matmul_desc.reset(new dnnl::matmul::desc(key.inp0->getDnnlDesc(),
                                                       key.inp1->getDnnlDesc(),
                                                       key.out->getDnnlDesc()));
        }

        DnnlDesriptor desc(matmul_desc);
        primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, key.attr);
        matmul::primitive_desc prim_desc;

        while (static_cast<bool>(itpd))  {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                prim_desc = itpd.get();
                break;
            }
            if (!itpd.next_impl())
                return matmul();
        }
        return matmul(prim_desc);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    prim = result.first;

    auto pd = prim.get_primitive_desc();
    auto scratchpadMem = getScratchPadMem(pd);

    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
    primArgs[DNNL_ARG_SRC_0] = src0MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src1MemPtr->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();
    if (withBiases)
        primArgs[DNNL_ARG_BIAS] = getParentEdgeAt(2)->getMemoryPtr()->GetPrimitive();

    appendPostOpArgs(*attr, primArgs, postOpsArgs);
}

void MatMul::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

const std::vector<impl_desc_type>& MatMul::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::brgemm_avx512_amx,
            impl_desc_type::brgemm_avx512,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx,
            impl_desc_type::gemm_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm,
            impl_desc_type::jit_gemm,
            impl_desc_type::jit_uni_dw,
            impl_desc_type::jit_uni_1x1,
            impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw,
            impl_desc_type::jit_avx512_1x1,
            impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw,
            impl_desc_type::jit_avx2_1x1,
            impl_desc_type::jit_avx2,
            impl_desc_type::jit_avx_dw,
            impl_desc_type::jit_avx_1x1,
            impl_desc_type::jit_avx,
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
