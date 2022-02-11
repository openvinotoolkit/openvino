// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_fullyconnected_node.h"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_fake_quantize_node.h"
#include "ngraph_transformations/op/fully_connected.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <vector>
#include <mkldnn_extension_utils.h>
#include <mkldnn.hpp>
#include "utils/general_utils.h"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"
#include <common/primitive_hashing_utils.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace {

struct FCKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;
    mkldnn::primitive_attr attr;
    impl_desc_type implType;

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
             implType == rhs.implType;
    return retVal;
}

} // namespace

bool MKLDNNFullyConnectedNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
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

MKLDNNFullyConnectedNode::MKLDNNFullyConnectedNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache), withBiases(false) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "FullyConnected node with name '" + getName() + "'";

        withBiases = inputShapes.size() == 3;
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

std::vector<memory::format_tag> MKLDNNFullyConnectedNode::getAvailableFormatsForDims(const Shape &dims) const {
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

VectorDims MKLDNNFullyConnectedNode::makeDummyInputDims() const {
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

VectorDims MKLDNNFullyConnectedNode::makeDummyOutputDims(const VectorDims& inDims) const {
    std::vector<Shape> inShapes = {Shape(inDims), getInputShapeAtPort(WEIGHTS_ID)};
    if (inputShapes.size() > 2) {
        inShapes.emplace_back(getInputShapeAtPort(BIAS_ID));
    }
    return shapeInferGeneric(inShapes).front();
}

void MKLDNNFullyConnectedNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2 && getParentEdges().size() != 3)
        IE_THROW() << errorPrefix << " has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW()<< errorPrefix << " has incorrect number of output edges";

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(DATA_ID));
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(DATA_ID));

    if (inputDataType == memory::data_type::f32) {
        outputDataType = memory::data_type::f32;
    }

    if (!fusedWith.empty()) {
        outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0));
    }
    auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(WEIGHTS_ID));

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
        auto in_candidate = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(inDims), inputDataType, format);
        auto out_candidate = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(outDims), outputDataType, mkldnn::memory::format_tag::any);

        createDescriptorInternal(in_candidate, out_candidate);
    }
}

void MKLDNNFullyConnectedNode::prepareParams() {
    auto srcMemPtr = getParentEdgesAtPort(0)[0]->getMemoryPtr();
    auto wghMemPtr = getParentEdgesAtPort(1)[0]->getMemoryPtr();
    auto dstMemPtr = getChildEdgesAtPort(0)[0]->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory hasn't been allocated.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory hasn't been allocated.";
    if (!wghMemPtr || !wghMemPtr->isAllocated())
        IE_THROW() << "Weight memory hasn't been allocated.";
    MKLDNNMemoryPtr biasMemPtr = nullptr;
    if (withBiases) {
        biasMemPtr = getParentEdgesAtPort(2)[0]->getMemoryPtr();
        if (!biasMemPtr || !biasMemPtr->isAllocated())
            IE_THROW() << "Input memory hasn't been allocated.";
    }

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";

    AttrPtr attr = std::make_shared<mkldnn::primitive_attr>();
    setPostOps(*attr, dstMemPtr->getStaticDims());

    DnnlMemoryDescCPtr weightDesc = wghMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr biasDesc = nullptr;
    if (biasMemPtr) {
        biasDesc = biasMemPtr->GetDescWithType<DnnlMemoryDesc>();
    }

    DnnlMemoryDescCPtr inDesc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>();
    DnnlMemoryDescCPtr outDesc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>();

    FCKey key = {inDesc,
                 weightDesc,
                 biasDesc,
                 outDesc,
                 *attr,
                 selected_pd->getImplementationType()};

    auto engine = getEngine();

    auto builder = [&engine](const FCKey& key) -> std::shared_ptr<mkldnn::primitive> {
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

        std::shared_ptr<mkldnn::inner_product_forward::desc> fcDsc;
        if (key.bias) {
            fcDsc = std::make_shared<mkldnn::inner_product_forward::desc>(mkldnn::prop_kind::forward_scoring,
                                                                          inDesc,
                                                                          key.inp1->getDnnlDesc(),
                                                                          key.bias->getDnnlDesc(),
                                                                          outDesc);
        } else {
            fcDsc = std::make_shared<mkldnn::inner_product_forward::desc>(mkldnn::prop_kind::forward_scoring,
                                                                          inDesc,
                                                                          key.inp1->getDnnlDesc(),
                                                                          outDesc);
        }
        MKLDNNDescriptor desc(fcDsc);
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

        return std::make_shared<inner_product_forward>(prim_desc);
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

    prim = result.first;

    primArgs[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
    primArgs[DNNL_ARG_WEIGHTS] = wghMemPtr->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dstMemPtr->GetPrimitive();

    if (withBiases) {
        primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
    }

    appendPostOpArgs(*attr, primArgs, binaryPostOpsArgs);

    auto reshapeMemory = [this](int argType) {
        auto param = primArgs.find(argType);
        if (param != primArgs.end()) {
            auto oldMem = param->second;
            auto dims = oldMem.get_desc().dims();
            if (dims.size() == 3) {
                std::vector<dnnl::memory::dim> normalizedDims({dims[0] * dims[1], dims[2]});
                mkldnn::memory::desc newMemDesc(oldMem.get_desc().reshape(normalizedDims));
                mkldnn::memory newMem(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());
                primArgs.at(argType) = newMem;
            }
        }
    };
    reshapeMemory(DNNL_ARG_SRC);
    reshapeMemory(DNNL_ARG_DST);
}

void MKLDNNFullyConnectedNode::setDynamicBatchLim(int lim) {
    dynBatchLim = lim;

    auto setBatchPrimArgs = [this](int argType, const mkldnn::memory& oldMem) {
        mkldnn::memory::desc newMemDesc(oldMem.get_desc());
        newMemDesc.data.dims[0] = batchToProcess();
        newMemDesc.data.padded_dims[0] = batchToProcess();
        auto dims = newMemDesc.dims();

        if (dims.size() == 3) {
            std::vector<dnnl::memory::dim> normalizedDims({dims[0] * dims[1], dims[2]});
            newMemDesc = newMemDesc.reshape(normalizedDims);
        }

        primArgs.at(argType) = mkldnn::memory(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());
    };

    setBatchPrimArgs(DNNL_ARG_SRC, getParentEdgesAtPort(0)[0]->getMemory().GetPrimitive());
    setBatchPrimArgs(DNNL_ARG_DST, getChildEdgesAtPort(0)[0]->getMemory().GetPrimitive());
}

void MKLDNNFullyConnectedNode::execute(mkldnn::stream strm) {
    if (prim) {
        // in cases parameter -> FullyConnected or dynamic shapes
        // we keep old pointer to data in primArgs on second iteration with same input shapes
        auto updateMemoryPtr = [this](int argType) {
            auto param = primArgs.find(argType);
            if (param != primArgs.end()) {
                if (argType == DNNL_ARG_SRC && getInputShapeAtPort(DATA_ID).getRank() == 3) {
                    primArgs.at(argType).set_data_handle(getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
                }
                if (argType == DNNL_ARG_DST && getOutputShapeAtPort(0).getRank() == 3) {
                    primArgs.at(argType).set_data_handle(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetData());
                }
            }
        };

        updateMemoryPtr(DNNL_ARG_SRC);
        updateMemoryPtr(DNNL_ARG_DST);

        (*prim).execute(strm, primArgs);
    }
}

void MKLDNNFullyConnectedNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNFullyConnectedNode::canFuse(const MKLDNNNodePtr& node) const {
    return canFuseSimpleOperation(node);
}

void MKLDNNFullyConnectedNode::setPostOps(mkldnn::primitive_attr &attr, const VectorDims &dims, bool initWeights) {
    mkldnn::post_ops ops;

    auto getBinPostOpShape = [&](){
        const size_t binaryShapeRank = getOutputShapeAtPort(0).getRank() == 3 ? 2 : getOutputShapeAtPort(0).getRank();
        VectorDims binaryShape(binaryShapeRank, 1);
        const size_t channelAxis = getFusingAxis();
        // always use 1 as channelAxis for binary Shape, since oneDNN primitive is actually always 2D
        binaryShape[1] = dims[channelAxis];

        return binaryShape;
    };

    for (auto &node : fusedWith) {
        if (auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get())) {
            fakeQuantizeNode->appendBinPostOps(ops, getBinPostOpShape(), binaryPostOpsArgs);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            if (eltwiseNode->getMKLDNNAlgorithm() != mkldnn::algorithm::undef) {
                eltwiseNode->appendPostOps(ops, dims);
            } else {
                eltwiseNode->appendBinPostOps(ops, getBinPostOpShape(), binaryPostOpsArgs);
            }
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

bool MKLDNNFullyConnectedNode::created() const {
    return getType() == FullyConnected;
}

const std::vector<impl_desc_type>& MKLDNNFullyConnectedNode::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
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

    // WA: brgemm kernel contains bug that may lead to segfault in case of added post-ops and unaligned number of channels
    const size_t simdWidth = 16;
    auto inputDims = getInputShapeAtPort(DATA_ID).getDims();
    if (inputDims.back() != Shape::UNDEFINED_DIM && inputDims.back() % simdWidth == 0) {
        priorities.insert(priorities.begin() + 1, impl_desc_type::brgemm_avx512_amx);
        priorities.insert(priorities.begin() + 2, impl_desc_type::brgemm_avx512);
    }

    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

MKLDNNNode::AttrPtr MKLDNNFullyConnectedNode::initPrimitiveAttr() {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, outDims);

    return attr;
}

// WA: creation DnnlMemoryDesc with format == any is prohibited
// so we create mkldnn::memory::desc directly
// we need specific method and can't remove createDescriptor from base class because its used into initDescriptor
void MKLDNNFullyConnectedNode::createDescriptorInternal(const mkldnn::memory::desc &inputDesc,
                                                        const mkldnn::memory::desc &outputDesc) {
    auto in_candidate = inputDesc;
    auto out_candidate = outputDesc;

    mkldnn::memory::data_type wdt = in_candidate.data_type();
    mkldnn::memory::data_type bdt = out_candidate.data_type();
    if (in_candidate.data_type() == mkldnn::memory::data_type::bf16) {
        bdt = mkldnn::memory::data_type::f32;
    } else if (in_candidate.data_type() == mkldnn::memory::data_type::u8 || in_candidate.data_type() == mkldnn::memory::data_type::s8) {
        wdt = memory::data_type::s8;
        if (withBiases)
            bdt = MKLDNNExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(BIAS_ID));
    }

    if (in_candidate.dims().size() == 3) {
        auto inDims = in_candidate.dims();
        auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};
        in_candidate = mkldnn::memory::desc(normalizedInDims, in_candidate.data_type(),
                                         MKLDNNExtensionUtils::GetPlainFormatByRank(normalizedInDims.size()));
    }

    if (out_candidate.dims().size() == 3) {
        auto outDims = out_candidate.dims();
        auto normalizedOutDims = { outDims[0] * outDims[1], outDims[2] };
        out_candidate = mkldnn::memory::desc(normalizedOutDims, out_candidate.data_type(),
                                         MKLDNNExtensionUtils::GetPlainFormatByRank(normalizedOutDims.size()));
    }

    mkldnn::memory::desc wgh_candidate(MKLDNNExtensionUtils::convertToDnnlDims(getInputShapeAtPort(WEIGHTS_ID).getStaticDims()),
                                       wdt, mkldnn::memory::format_tag::any);

    if (withBiases) {
        mkldnn::memory::desc bias_candidate(MKLDNNExtensionUtils::convertToDnnlDims(getInputShapeAtPort(BIAS_ID).getStaticDims()), bdt,
                                            mkldnn::memory::format_tag::any);
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                bias_candidate, out_candidate)));
        descs.push_back(desc);
    } else {
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                out_candidate)));
        descs.push_back(desc);
    }
}

void MKLDNNFullyConnectedNode::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
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

void MKLDNNFullyConnectedNode::initSupportedPrimitiveDescriptors() {
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
                if (supportsUndefStridesAndOffset()) {
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

std::shared_ptr<MemoryDesc> MKLDNNFullyConnectedNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : primitive_desc_it.src_desc(idx);

    if (getInputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(MKLDNNExtensionUtils::DataTypeToIEPrecision(
            static_cast<mkldnn::memory::data_type>(desc.data.data_type)), getInputShapeAtPort(idx));
    }

    if (getInputShapeAtPort(idx).isDynamic()) {
        return MKLDNNExtensionUtils::makeUndefinedDesc(desc, getInputShapeAtPort(idx));
    }

    return MKLDNNExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> MKLDNNFullyConnectedNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = primitive_desc_it.dst_desc(idx);

    if (getOutputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(MKLDNNExtensionUtils::DataTypeToIEPrecision(
            static_cast<mkldnn::memory::data_type>(desc.data.data_type)), getOutputShapeAtPort(idx));
    }

    if (getOutputShapeAtPort(idx).isDynamic()) {
        return MKLDNNExtensionUtils::makeUndefinedDesc(desc, getOutputShapeAtPort(idx));
    }

    return MKLDNNExtensionUtils::makeDescriptor(desc);
}

InferenceEngine::Precision MKLDNNFullyConnectedNode::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    // Don't take bias precision into account
    size_t inputsNumLimit = 2;
    for (size_t i = 0; i < std::min(getParentEdges().size(), inputsNumLimit); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNFullyConnectedNode, FullyConnected);
