// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.h"
#include "eltwise.h"
#include "input.h"
#include "fake_quantize.h"
#include "input.h"
#include "reorder.h"
#include "ngraph_transformations/op/fully_connected.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <vector>
#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>
#include "utils/general_utils.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"
#include <common/primitive_hashing_utils.hpp>
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#include "onednn/dnnl.h"
#include "cpu/x64/cpu_isa_traits.hpp"

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct FCKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    dnnl::primitive_attr attr;
    impl_desc_type implType;
    bool useConv1x1;

    size_t hash() const;
    bool operator==(const FCKey& rhs) const;
};

size_t FCKey::hash() const {
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
    seed = hash_combine(seed, useConv1x1);
    return seed;
}

bool FCKey::operator==(const FCKey &rhs) const {
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
             implType == rhs.implType && useConv1x1 == rhs.useConv1x1;
    return retVal;
}

class FCShapeInfer : public ShapeInferEmptyPads {
public:
    FCShapeInfer(size_t outPut_rank) : out_rank(outPut_rank) {}
    std::vector<VectorDims> infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const VectorDims& activationShape = input_shapes[0].get();
        const VectorDims& weightShape = input_shapes[1].get();
        size_t activationRank = activationShape.size();
        size_t channelRank = weightShape.size() - 1;

        // activation   weight    output_shape
        // NCHW         CoCHW     NCo
        // TNC          CoC       TNCo
        // NC           CoC       NCo
        VectorDims outputShape(out_rank, 1);
        // set Co
        outputShape.back() = weightShape[0];
        // set batch dims
        size_t batchRank = activationRank - channelRank;
        size_t startIdx = out_rank - batchRank - 1;
        for (size_t i = 0; i < batchRank; i++) {
            outputShape[i + startIdx] = activationShape[i];
        }

        return {outputShape};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    size_t out_rank = 0;
};

class FCShapeInferFactory : public ShapeInferFactory {
public:
    FCShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<FCShapeInfer>(m_op->get_output_partial_shape(0).rank().get_length());
    }

private:
    std::shared_ptr<const ngraph::Node> m_op;
};

} // namespace

bool FullyConnected::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto fc = std::dynamic_pointer_cast<const FullyConnectedNode>(op);
        if (!fc) {
            errorMessage = "Only legacy FullyConnected operation is supported";
            return false;
        }
        if (fc->get_input_size() == 3 && std::dynamic_pointer_cast<const ngraph::opset1::Constant>(fc->get_input_node_shared_ptr(BIAS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'bias' input is supported";
            return false;
        }
        const auto inRank = fc->get_input_partial_shape(DATA_ID).size();
        const auto weightRank = fc->get_input_partial_shape(WEIGHTS_ID).size();
        if (!one_of(inRank, 2, 3, 4)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank);
            return false;
        }
        if ((one_of(inRank, 2, 3) && weightRank != 2) || (inRank == 4 && weightRank != 4)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inRank) +
                           " and 'weight' input with rank: " + std::to_string(weightRank);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

FullyConnected::FullyConnected(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, FCShapeInferFactory(op)), withBiases(false) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "FullyConnected node with name '" + getName() + "'";

        withBiases = inputShapes.size() == 3;

        if (context->getConfig().fcSparseWeiDecompressionRate < 1.0f)
            minSparseRate = context->getConfig().fcSparseWeiDecompressionRate;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

std::vector<memory::format_tag> FullyConnected::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nChw8c, memory::format_tag::nChw16c, memory::format_tag::nhwc, memory::format_tag::nchw};
    else if (dims.getRank() == 5)
        return {memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c, memory::format_tag::ndhwc, memory::format_tag::ncdhw};
    return {memory::format_tag::any};
}

VectorDims FullyConnected::makeDummyInputDims() const {
    const auto& inShape = getInputShapeAtPort(DATA_ID);
    const auto& weightDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();

    if (inMinDims.size() == 3) {
        inMinDims.back() = weightDims.back();
        inMaxDims.back() = weightDims.back();
    } else {
        for (size_t i = 1; i < inMinDims.size(); i++) {
            inMinDims[i] = weightDims[i];
            inMaxDims[i] = weightDims[i];
        }
    }
    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

VectorDims FullyConnected::makeDummyOutputDims(const VectorDims& inDims) const {
    std::vector<Shape> inShapes = {Shape(inDims), getInputShapeAtPort(WEIGHTS_ID)};
    if (inputShapes.size() > 2) {
        inShapes.emplace_back(getInputShapeAtPort(BIAS_ID));
    }
    return shapeInferGeneric(inShapes).front();
}

void FullyConnected::getSupportedDescriptors() {
    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW()<< errorPrefix << " has incorrect number of output edges";

    useSparseWeights = useSparseWeightsDecompression();

    auto inputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(DATA_ID));
    outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(DATA_ID));

    if (inputDataType == memory::data_type::f32) {
        outputDataType = memory::data_type::f32;
    }

    if (!fusedWith.empty()) {
        outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    auto weightsDataType = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(WEIGHTS_ID));

    //  We have to extend gemm_x8s8s32x_inner_product_fwd_t from oneDNN to support BF16 output data type
    if ((!one_of(inputDataType , memory::data_type::u8, memory::data_type::s8) || weightsDataType != memory::data_type::s8)
            && inputDataType != memory::data_type::bf16) {
        inputDataType = outputDataType = memory::data_type::f32;
    }

    if (one_of(inputDataType , memory::data_type::u8, memory::data_type::s8)
        && outputDataType == memory::data_type::bf16) {
        outputDataType = memory::data_type::f32;
    }

    if (inputDataType == memory::data_type::bf16
        && one_of(outputDataType , memory::data_type::u8, memory::data_type::s8)) {
        outputDataType = memory::data_type::bf16;
    }

    inDims = isDynamicNode() ? makeDummyInputDims() : getInputShapeAtPort(DATA_ID).getStaticDims();
    outDims = isDynamicNode() ? makeDummyOutputDims(inDims) : getOutputShapeAtPort(0).getStaticDims();

    for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
        auto in_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(inDims), inputDataType, format);
        auto out_candidate = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(outDims), outputDataType, dnnl::memory::format_tag::any);

        createDescriptorInternal(in_candidate, out_candidate);
    }
}

void FullyConnected::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory hasn't been allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory hasn't been allocated.";
    MemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        if (!biasMemPtr || !biasMemPtr->isAllocated())
            IE_THROW() << "Input memory hasn't been allocated.";
    }

    NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    AttrPtr attr = std::make_shared<dnnl::primitive_attr>();
    setPostOps(*attr, dstMemPtr->getStaticDims());
    (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

    DnnlMemoryDescPtr weightDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightDescIP);
    DnnlMemoryDescCPtr biasDesc = nullptr;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
    }

    DnnlMemoryDescCPtr inDesc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    useConv1x1 = canBeExecutedInConv1x1();
    FCKey key = {inDesc,
                 weightDesc,
                 biasDesc,
                 outDesc,
                 *attr,
                 implementationTypeIP,
                 useConv1x1};

    auto engine = getEngine();

    auto builder = [&engine](const FCKey& key) -> executorPtr {
        executorPtr execPtr = nullptr;
        if (key.useConv1x1) {
            auto desc = createDescriptorInternalForConv(key.inp0, key.inp1, key.bias, key.out);
            primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, key.attr);
            convolution_forward::primitive_desc prim_desc;

            while (static_cast<bool>(itpd))  {
                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

                if (impl_type == brgconv_avx512_1x1) {
                    prim_desc = itpd.get();
                    break;
                }
                if (!itpd.next_impl()) {
                    break;
                }
            }

            if (prim_desc) {
                execPtr = std::make_shared<ExecutorConv1x1>(prim_desc);
            }
        }
        // fallback
        if (!execPtr) {
            auto inDesc = key.inp0->getDnnlDesc();
            if (inDesc.dims().size() == 3) {
                auto inDims = inDesc.dims();
                auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};
                inDesc = inDesc.reshape(normalizedInDims);
            }

            auto outDesc = key.out->getDnnlDesc();
            if (outDesc.dims().size() == 3) {
                auto outDims = outDesc.dims();
                auto normalizedOutDims = { outDims[0] * outDims[1], outDims[2] };
                outDesc = outDesc.reshape(normalizedOutDims);
            }

            std::shared_ptr<dnnl::inner_product_forward::desc> fcDsc;
            if (key.bias) {
                fcDsc = std::make_shared<dnnl::inner_product_forward::desc>(dnnl::prop_kind::forward_scoring,
                                                                            inDesc,
                                                                            key.inp1->getDnnlDesc(),
                                                                            key.bias->getDnnlDesc(),
                                                                            outDesc);
            } else {
                fcDsc = std::make_shared<dnnl::inner_product_forward::desc>(dnnl::prop_kind::forward_scoring,
                                                                            inDesc,
                                                                            key.inp1->getDnnlDesc(),
                                                                            outDesc);
            }
            DnnlDesriptor desc(fcDsc);
            primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine, key.attr);
            inner_product_forward::primitive_desc prim_desc;

            while (static_cast<bool>(itpd))  {
                impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

                if (impl_type == key.implType) {
                    prim_desc = itpd.get();
                    break;
                }
                if (!itpd.next_impl()) {
                    return nullptr;
                }
            }

            execPtr = std::make_shared<ExecutorInnerProduct>(prim_desc);
        }
        return execPtr;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    auto prevExecPtr = execPtr;
    execPtr = result.first;

    if (execPtr) {
        // no executor yet or shapes changed
        if (!prevExecPtr || prevExecPtr->getSrcDesc() != execPtr->getSrcDesc()) {
            auto oldMem = srcMemPtr->GetPrimitive();
            // fast path: wanted is same with parent node output, typical is static shape with inner product
            if (execPtr->getSrcDesc() == inDesc->getDnnlDesc()) {
                primArgs[DNNL_ARG_SRC] = std::move(oldMem);
            } else {
                primArgs[DNNL_ARG_SRC] = dnnl::memory(execPtr->getSrcDesc(), oldMem.get_engine(), oldMem.get_data_handle());
            }
        }
        if (!prevExecPtr || prevExecPtr->getDstDesc() != execPtr->getDstDesc()) {
            auto oldMem = dstMemPtr->GetPrimitive();
            if (execPtr->getDstDesc() == outDesc->getDnnlDesc()) {
                primArgs[DNNL_ARG_DST] = std::move(oldMem);
            } else {
                primArgs[DNNL_ARG_DST] = dnnl::memory(execPtr->getDstDesc(), oldMem.get_engine(), oldMem.get_data_handle());
            }
        }
        if (!prevExecPtr || prevExecPtr->getWeightDesc() != execPtr->getWeightDesc()) {
            primArgs[DNNL_ARG_WEIGHTS] = prepareWeightMemory(DnnlExtensionUtils::makeDescriptor(execPtr->getWeightDesc()))->GetPrimitive();
        }
        // changed shapes may also cause the kernel type changed
        selected_pd->setImplementationType(execPtr->getImplementationType());
        // WA: We update implType to know whether weights decompression was used inside the kernel
        if (selected_pd->getImplementationType() == ov::intel_cpu::brgemm_avx512_amx && useSparseWeights) {
            selected_pd->setImplementationType(ov::intel_cpu::brgemm_sparse_avx512_amx);
        }
        // maybe expected 1x1 conv is not created, update the flag depends on the real type
        useConv1x1 = execPtr->getImplementationType() == brgconv_avx512_1x1;

        if (withBiases) {
            primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
        }

        appendPostOpArgs(*attr, primArgs, postOpsArgs);

        auto pd = execPtr->getPrimitiveDesc();
        auto scratchpadMem = getScratchPadMem(pd);
        primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
#ifdef CPU_DEBUG_CAPS
        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
            DEBUG_LOG("verbose##", getName(), "##", pd->info(), "\n");
        }
#endif
    } else {
        IE_THROW() << "Executor is not created for node " << getName() << ".";
    }
}

void FullyConnected::setDynamicBatchLim(int lim) {
    if (!execPtr) {
        IE_THROW() << "Can't set dynamic batch for FullyConnected node with name: " << getName() << ", because executor is not compiled";
    }
    if (execPtr->needReordering()) {
        IE_THROW() << "Can't execute FullyConnected node with dynamic batch via executor with reorders";
    }

    auto setBatchPrimArgs = [this](int argType, const dnnl::memory& oldMem) {
        dnnl::memory::desc newMemDesc(oldMem.get_desc());
        newMemDesc.data.dims[0] = batchToProcess();
        newMemDesc.data.padded_dims[0] = batchToProcess();
        auto dims = newMemDesc.dims();

        if (dims.size() == 3) {
            std::vector<dnnl::memory::dim> normalizedDims({dims[0] * dims[1], dims[2]});
            newMemDesc = newMemDesc.reshape(normalizedDims);
        }

        primArgs.at(argType) = dnnl::memory(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());
    };

    if (useConv1x1) {
        Node::setDynamicBatchLim(lim);
    } else {
        dynBatchLim = lim;
        setBatchPrimArgs(DNNL_ARG_SRC, getParentEdgesAtPort(0)[0]->getMemory().GetPrimitive());
        setBatchPrimArgs(DNNL_ARG_DST, getChildEdgesAtPort(0)[0]->getMemory().GetPrimitive());
    }
}

void FullyConnected::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute FullyConnected node with name: " << getName() << ", because executor is not compiled";
    }

    // in cases parameter -> FullyConnected or dynamic shapes
    // we keep old pointer to data in primArgs on second iteration with same input shapes
    auto updateMemoryPtr = [this](int argType) {
        auto param = primArgs.find(argType);
        if (param != primArgs.end()) {
            if (argType == DNNL_ARG_SRC && (getInputShapeAtPort(DATA_ID).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
            }
            if (argType == DNNL_ARG_DST && (getOutputShapeAtPort(0).getRank() == 3 || useConv1x1)) {
                primArgs.at(argType).set_data_handle(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
            }
        }
    };

    updateMemoryPtr(DNNL_ARG_SRC);
    updateMemoryPtr(DNNL_ARG_DST);

    execPtr->exec(primArgs, strm);
}

void FullyConnected::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool FullyConnected::canFuse(const NodePtr& node) const {
    return canFuseSimpleOperation(node);
}

void FullyConnected::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims_ext, bool initWeights) {
    dnnl::post_ops ops;

    // accoridng to https://oneapi-src.github.io/oneDNN/dev_guide_inner_product.html
    // oneDNN inner product primitive's input & output tensors are always 2D:
    //   input: [N, IC]  weight: [OC, IC]   bias: [OC]   output:[N,OC]
    //
    // when input output tensors have spatial dimensions, they are flattened to 2D.
    // and following type of MatMul will be converted into FullyConnected inside CPU plugin:
    //    2D:   [X,Y] [Y,Z] =>   [X,Z]   with    N=X,IC=Y,OC=Z
    //    3D: [B,X,Y] [Y,Z] => [B,X,Z]   with  N=B*X,IC=Y,OC=Z

    VectorDims dims;
    if (dims_ext.size() == 2) {
        // 2D
        dims = dims_ext;
    } else if (dims_ext.size() == 3) {
        // 3D
        dims.push_back(dims_ext[0] * dims_ext[1]);
        dims.push_back(dims_ext[2]);
    } else {
        IE_THROW() << "Unexpected rank(" << dims_ext.size() << ") for output tensor of node: " << getName();
    }

    bool isINT8 = getOriginalInputPrecisionAtPort(WEIGHTS_ID) == Precision::U8 ||
                  getOriginalInputPrecisionAtPort(WEIGHTS_ID) == Precision::I8;

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8);

    for (int i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

bool FullyConnected::created() const {
    return getType() == Type::FullyConnected;
}

const std::vector<impl_desc_type>& FullyConnected::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::brgemm_sparse_avx512_amx,
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

Node::AttrPtr FullyConnected::initPrimitiveAttr() {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, outDims);

    return attr;
}

// WA: creation DnnlMemoryDesc with format == any is prohibited
// so we create dnnl::memory::desc directly
// we need specific method and can't remove createDescriptor from base class because its used into initDescriptor
void FullyConnected::createDescriptorInternal(const dnnl::memory::desc &inputDesc,
                                                        const dnnl::memory::desc &outputDesc) {
    auto in_candidate = inputDesc;
    auto out_candidate = outputDesc;

    dnnl::memory::data_type wdt = in_candidate.data_type();
    dnnl::memory::data_type bdt = out_candidate.data_type();
    if (in_candidate.data_type() == dnnl::memory::data_type::bf16) {
        bdt = dnnl::memory::data_type::f32;
    } else if (in_candidate.data_type() == dnnl::memory::data_type::u8 || in_candidate.data_type() == dnnl::memory::data_type::s8) {
        wdt = memory::data_type::s8;
        if (withBiases)
            bdt = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(BIAS_ID));
    }

    if (in_candidate.dims().size() == 3) {
        auto inDims = in_candidate.dims();
        auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};
        in_candidate = dnnl::memory::desc(normalizedInDims, in_candidate.data_type(),
                                         DnnlExtensionUtils::GetPlainFormatByRank(normalizedInDims.size()));
    }

    if (out_candidate.dims().size() == 3) {
        auto outDims = out_candidate.dims();
        auto normalizedOutDims = { outDims[0] * outDims[1], outDims[2] };
        out_candidate = dnnl::memory::desc(normalizedOutDims, out_candidate.data_type(),
                                         DnnlExtensionUtils::GetPlainFormatByRank(normalizedOutDims.size()));
    }

    // We need to explicitly specify the memory descriptor to use sparse weights decompression
    dnnl::memory::desc wgh_candidate;
    if (useSparseWeights) {
        wgh_candidate = { DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                wdt, memory::desc::packed(nnzCount) };
    } else {
        wgh_candidate = { DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                        wdt, dnnl::memory::format_tag::any };
    }
    if (withBiases) {
        dnnl::memory::desc bias_candidate(DnnlExtensionUtils::convertToDnnlDims(getInputShapeAtPort(BIAS_ID).getStaticDims()), bdt,
                                            dnnl::memory::format_tag::any);
        DnnlDesriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                bias_candidate, out_candidate)));
        descs.push_back(desc);
    } else {
        DnnlDesriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                out_candidate)));
        descs.push_back(desc);
    }
}

void FullyConnected::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                                const std::vector<MemoryDescPtr> &outputDesc) {
    MemoryDescPtr inpDesc;
    if (inputDesc[0]->isDefined()) {
        inpDesc = inputDesc[0];
    } else {
        inpDesc = inputDesc[0]->cloneWithNewDims(inDims);
    }

    MemoryDescPtr outDesc;
    if (outputDesc[0]->isDefined()) {
        outDesc = outputDesc[0];
    } else {
        outDesc = outputDesc[0]->cloneWithNewDims(outDims);
    }
    createDescriptorInternal(MemoryDescUtils::convertToDnnlMemoryDesc(inpDesc)->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(outDesc)->getDnnlDesc());
}

void FullyConnected::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine());
        while (static_cast<bool>(itpd)) {
            // 3D FC requires implicit reshape so strides should be defined
            auto supportsUndefStridesAndOffset = [&]() {
                return getOutputShapeAtPort(0).getRank() == 2;
            };

            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace(-1);
                portConfig.constant(false);
                auto desc = getSrcMemDesc(itpd, i);
                if (supportsUndefStridesAndOffset() && !(i == WEIGHTS_ID && useSparseWeights)) {
                    portConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                } else {
                    portConfig.setMemDesc(desc);
                }
                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace(canBeInPlace() ? 0 : -1);
                portConfig.constant(false);
                auto desc = getDstMemDesc(itpd, i);
                if (supportsUndefStridesAndOffset()) {
                    portConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                } else {
                    portConfig.setMemDesc(desc);
                }
                config.outConfs.push_back(portConfig);
            }

            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

std::shared_ptr<MemoryDesc> FullyConnected::getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : primitive_desc_it.src_desc(idx);

    if (getInputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(DnnlExtensionUtils::DataTypeToIEPrecision(
            static_cast<dnnl::memory::data_type>(desc.data.data_type)), getInputShapeAtPort(idx));
    }

    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> FullyConnected::getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = primitive_desc_it.dst_desc(idx);

    if (getOutputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(DnnlExtensionUtils::DataTypeToIEPrecision(
            static_cast<dnnl::memory::data_type>(desc.data.data_type)), getOutputShapeAtPort(idx));
    }

    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }

    return DnnlExtensionUtils::makeDescriptor(desc);
}

InferenceEngine::Precision FullyConnected::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void FullyConnected::initOptimalPrimitiveDescriptor() {
    Node::initOptimalPrimitiveDescriptor();
    auto selectedPD = getSelectedPrimitiveDescriptor();
    implementationTypeIP = selectedPD->getImplementationType();
    // if convolution selected the reorder for ip is useless. Will do the reoder for ip in prepareParams
    auto constParent = getParentEdgeAt(1)->getParent();
    auto selectedParentPD = constParent->getSelectedPrimitiveDescriptor();
    auto config = selectedPD->getConfig();
    weightDescIP = config.inConfs[1].getMemDesc();
    config.inConfs[1].setMemDesc(selectedParentPD->getConfig().outConfs[0].getMemDesc());
    selectedPD->setConfig(config);
}

DnnlDesriptor FullyConnected::createDescriptorInternalForConv(DnnlMemoryDescCPtr inputDescPtr,
                                                              DnnlMemoryDescCPtr weightDescPtr,
                                                              DnnlMemoryDescCPtr biasDescPtr,
                                                              DnnlMemoryDescCPtr outputDescPtr) {
    const dnnl::memory::desc &inputDesc = inputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &outputDesc = outputDescPtr->getDnnlDesc();
    const dnnl::memory::desc &weightDesc = weightDescPtr->getDnnlDesc();

    // make a fake shape: N, IC, W
    auto inDims = inputDesc.dims();
    dnnl::memory::dims normalizedInDims;
    if (inDims.size() == 3) {
        normalizedInDims = {inDims[0], inDims[2], inDims[1]};
    } else if (inDims.size() == 2) {
        normalizedInDims = {dnnl::memory::dim{1}, inDims[1], inDims[0]};
    }
    auto convInDesc = dnnl::memory::desc(normalizedInDims, inputDesc.data_type(), memory::format_tag::nwc);

    // make a fake shape: N, OC, W
    auto outDims = outputDesc.dims();
    dnnl::memory::dims normalizedOutDims;
    if (outDims.size() == 3) {
        normalizedOutDims = { outDims[0], outDims[2], outDims[1]};
    } else if (outDims.size() == 2) {
        normalizedOutDims = { dnnl::memory::dim{1}, outDims[1], outDims[0]};
    }
    auto convOutDesc = dnnl::memory::desc(normalizedOutDims, outputDesc.data_type(), memory::format_tag::nwc);

    // make a fake shape: OC, IC, 1
    auto weightDims = weightDesc.dims();
    dnnl::memory::dims normalizedWeightDims;
    normalizedWeightDims = {static_cast<dnnl::memory::dim>(weightDims[0]),
                            static_cast<dnnl::memory::dim>(weightDims[1]),
                            dnnl::memory::dim{1}};
    auto convWeightDescAny = dnnl::memory::desc(normalizedWeightDims, weightDesc.data_type(), dnnl::memory::format_tag::any);

    std::shared_ptr<dnnl::convolution_forward::desc> desc;
    if (biasDescPtr) {
        desc = std::make_shared<dnnl::convolution_forward::desc>(prop_kind::forward_scoring, dnnl::algorithm::convolution_direct,
                                convInDesc, convWeightDescAny, biasDescPtr->getDnnlDesc(), convOutDesc,
                                dnnl::memory::dims{1},   // stride
                                dnnl::memory::dims{0},   // dilation
                                dnnl::memory::dims{0},   // paddingL
                                dnnl::memory::dims{0});  // paddingR
    } else {
        desc = std::make_shared<dnnl::convolution_forward::desc>(prop_kind::forward_scoring, dnnl::algorithm::convolution_direct,
                                convInDesc, convWeightDescAny, convOutDesc,
                                dnnl::memory::dims{1},   // stride
                                dnnl::memory::dims{0},   // dilation
                                dnnl::memory::dims{0},   // paddingL
                                dnnl::memory::dims{0});  // paddingR
    }

    return DnnlDesriptor(desc);
}

bool FullyConnected::canBeExecutedInConv1x1() const {
    bool retVal = false;
    const auto inRank = getInputShapeAtPort(DATA_ID).getRank();
    const auto weightRank = getInputShapeAtPort(WEIGHTS_ID).getRank();
    // disable rank=4:
    // if layout is nhwc:
    //   A matrix: N * IC * H * W --> N * (IC*H*W), the M, N', K of matrix multiply will be:
    //   M = 1, K = (IC*H*W), when M = 1 it should not be efficient since acts as a vector multiply
    // if layout is nchw/nChw16c: brg1x1 not support. Although jit supports, it should have similar
    //   problems with the above.
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) &&
        getOriginalInputPrecisionAtPort(DATA_ID) == InferenceEngine::Precision::FP32 &&
        one_of(inRank, 2, 3) && weightRank == 2) {
        auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
        DnnlMemoryDescCPtr outDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();
        // brg convolution does not support stride
        if (outDesc->getDnnlDesc().data.offset0 == 0)
            retVal = true;
    }

    if (retVal) {
        auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
        const auto& srcDims = srcMemPtr->getStaticDims();
        auto weightMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
        const auto& weightDims = weightMemPtr->getStaticDims();
        // for original inner product semantics:
        //  when input is 2D tensor
        //    M in oneDNN will map to widthInConv
        //  when input is 3D tensor
        //    M in oneDNN will map to widthInConv*minibatch
        // currently nwc mapping in brg:
        //  when input is 2D tensor
        //    widthInConv will map to 'w', 'n' will be 1
        //  when input is 3D tensor
        //    widthInConv will map to 'w', 'n' will be minibatch
        Dim widthInConv, N, K;
        widthInConv = srcDims[inRank - 2];
        K = srcDims[inRank - 1];
        N = weightDims[0];

        if (!(widthInConv >= 2 && widthInConv <= 3136 &&
              K >= 96 && K <= 4096 &&
              N >= 96 && N <= K * 4))
            retVal = false;
    }

    return retVal;
}

FullyConnected::ExecutorInnerProduct::ExecutorInnerProduct(const dnnl::inner_product_forward::primitive_desc& pd) {
    execPrim = dnnl::inner_product_forward(pd);
}

FullyConnected::ExecutorConv1x1::ExecutorConv1x1(const dnnl::convolution_forward::primitive_desc& pd) {
    execPrim = dnnl::convolution_forward(pd);
}

MemoryPtr FullyConnected::prepareWeightMemory(DnnlMemoryDescPtr weightDesc) {
    if (!getParentEdgeAt(1)->getParent()->isConstant())
        IE_THROW() << "Weight input is not const for node " << getName() << ".";
    auto blob = getParentEdgeAt(1)->getMemoryPtr();
    if (!blob)
        IE_THROW() << "Cannot get const weights blob for node " << getName() << ".";

    auto constDnnlMemOutDesc = blob->GetDescWithType<DnnlMemoryDesc>();
    auto weightSrcDesc = constDnnlMemOutDesc->getDnnlDesc();
    weightSrcDesc = weightSrcDesc.reshape(weightDesc->getDnnlDesc().dims());
    auto create = [&] () {
        auto newSrcDesc = DnnlExtensionUtils::makeDescriptor(weightSrcDesc);

        Memory srcMemory{ getEngine() };
        srcMemory.Create(newSrcDesc, blob->GetData());

        MemoryPtr _ptr = std::make_shared<Memory>(getEngine());
        _ptr->Create(weightDesc);
        node::Reorder::reorderData(srcMemory, *_ptr, context->getParamsCache());

        return _ptr;
    };

    MemoryPtr ptr;
    const auto& format = weightDesc->serializeFormat();
    auto itr = privateWeightCache.find(format);
    if (privateWeightCache.end() != itr) {
        ptr = itr->second;
    } else {
        auto weightCache = context->getWeightsCache();
        if (weightCache != nullptr) {
            const std::string string_hash = getName() + "_" + format
                                            + "_" + std::to_string(blob->GetSize())
                                            + "_" + std::to_string(reinterpret_cast<uint64_t>(blob->GetData()));

            ptr = *weightCache->findOrCreate(string_hash, create);
        } else {
            ptr = create();
        }
        privateWeightCache[format] = ptr;
    }

    return ptr;
}

bool FullyConnected::useSparseWeightsDecompression() {
    // minSparseRate == 1 means that sparse feature is switched off
    if (minSparseRate == 1.f) {
        return false;
    }

    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx))
        return false;

    auto weiDims = getInputShapeAtPort(WEIGHTS_ID).getStaticDims();
    if (weiDims.size() != 2 || weiDims[0] % 64 != 0 || weiDims[1] % 64 != 0) {
        return false;
    }

    auto inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    auto weightsPrecision = getOriginalInputPrecisionAtPort(WEIGHTS_ID);
    if (!one_of(inputPrecision , Precision::U8, Precision::I8) || weightsPrecision != Precision::I8) {
        return false;
    }

    // calculate sparse rate
    const auto constNode = std::dynamic_pointer_cast<Input>(getParentEdgeAt(WEIGHTS_ID)->getParent());
    if (!constNode) {
        return false;
    }
    auto blb = constNode->getMemoryPtr();
    if (blb == nullptr)
        IE_THROW() << "Cannot get const blob for node " << getName() << ".";

    auto weightsData = reinterpret_cast<const int8_t*>(blb->GetPtr());
    auto elementsCount = blb->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    size_t zerosCounts = 0;
    for (int i = 0; i < elementsCount; i++) {
        if (weightsData[i] == 0) {
            zerosCounts++;
        }
    }
    nnzCount = elementsCount - zerosCounts;

    DEBUG_LOG(getName(), ", weightsData.size() = ", elementsCount, ", zerosCounts = ",
        zerosCounts, ", nnzCount = ", nnzCount);

    weiSparseRate = static_cast<float>(zerosCounts) / static_cast<float>(elementsCount);

    // [av] WA: there is no point in using sparse decompression when the sparse rate is low
    // todo: add heuristic
    if (minSparseRate < 0.5)
        minSparseRate = 0.5;

    DEBUG_LOG(getName(), " | sparse rate = ", weiSparseRate * 100, "%, min sparse rate = ",
        minSparseRate * 100, "%, use sparse weights = ", weiSparseRate >= minSparseRate);

    if (weiSparseRate < minSparseRate) {
        return false;
    }

    return true;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
