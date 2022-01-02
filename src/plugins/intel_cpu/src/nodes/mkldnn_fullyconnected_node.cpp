// Copyright (C) 2018-2021 Intel Corporation
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

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNFullyConnectedNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }

        const auto fc = std::dynamic_pointer_cast<const FullyConnectedNode>(op);
        if (!fc) {
            errorMessage = "Only legacy FullyConnected operation is supported";
            return false;
        }
        if (fc->get_input_size() == 3 && std::dynamic_pointer_cast<const ngraph::opset1::Constant>(fc->get_input_node_shared_ptr(BIAS_ID)) == nullptr) {
            errorMessage = "Only Constant operation on 'bias' input is supported";
            return false;
        }
        if (!one_of(fc->get_input_shape(DATA_ID).size(), 2, 3, 4)) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(fc->get_input_shape(DATA_ID).size());
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

        withBiases = op->get_input_size() == 3;
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

    const auto& inDims = getInputShapeAtPort(0).getStaticDims();
    const auto& outDims = getOutputShapeAtPort(0).getStaticDims();

    if (inDims.size() == 3) {
        weightsDims = InferenceEngine::SizeVector({static_cast<size_t>(outDims[2]), static_cast<size_t>(inDims[2])});
    } else {
        weightsDims.push_back(outDims[1]);
        for (int i = 1; i < inDims.size(); i++)
            weightsDims.push_back(inDims[i]);
    }
    biasesDims.push_back(weightsDims[0]);

    for (auto format : getAvailableFormatsForDims(getInputShapeAtPort(0))) {
        auto in_candidate = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(inDims), inputDataType, format);
        auto out_candidate = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(outDims), outputDataType, mkldnn::memory::format_tag::any);

        createDescriptorInternal(in_candidate, out_candidate);
    }
}

void MKLDNNFullyConnectedNode::createPrimitive() {
    if (prim)
        return;

    AttrPtr attr = initPrimitiveAttr();
    std::shared_ptr<inner_product_forward::primitive_desc> prim_desc;
    prim_desc = std::make_shared<inner_product_forward::primitive_desc>(
            createPrimitiveDescriptor<inner_product_forward::primitive_desc, inner_product_forward::desc>(*attr));

    prim.reset(new inner_product_forward(*prim_desc));

    auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
    if (withBiases)
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getParentEdgeAt(WEIGHTS_ID)->getMemory().GetPrimitive()},
                    {DNNL_ARG_BIAS, getParentEdgeAt(BIAS_ID)->getMemory().GetPrimitive()}, {DNNL_ARG_DST, dst}};
    else
        primArgs = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getParentEdgeAt(WEIGHTS_ID)->getMemory().GetPrimitive()}, {DNNL_ARG_DST, dst}};

    appendPostOpArgs(*attr, primArgs, binaryPostOpsArgs);
}

void MKLDNNFullyConnectedNode::execute(mkldnn::stream strm) {
    if (prim) {
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

        (*prim).execute(strm, primArgs);
    }
}

bool MKLDNNFullyConnectedNode::canFuse(const MKLDNNNodePtr& node) const {
    return canFuseSimpleOperation(node);
}

void MKLDNNFullyConnectedNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) {
    mkldnn::post_ops ops;

    auto getBinPostOpShape = [&](){
        const size_t binaryShapeRank = getOutputShapeAtPort(0).getRank() == 3 ? 2 : getOutputShapeAtPort(0).getRank();
        VectorDims binaryShape(binaryShapeRank, 1);
        const size_t channelAxis = getFusingAxis();
        // always use 1 as channelAxis for binary Shape, since oneDNN primitive is actually always 2D
        binaryShape[1] = getOutputShapeAtPort(0).getStaticDims()[channelAxis];

        return binaryShape;
    };

    for (auto &node : fusedWith) {
        if (auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get())) {
            fakeQuantizeNode->appendBinPostOps(ops, getBinPostOpShape(), binaryPostOpsArgs);
            continue;
        }

        if (auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get())) {
            // TODO [DS]: change to shape from memory
            if (eltwiseNode->getMKLDNNAlgorithm() != mkldnn::algorithm::undef) {
                eltwiseNode->appendPostOps(ops, getOutputShapeAtPort(0).getStaticDims());
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
    size_t simdWidth = 16;
    auto inputDims = inputShapes[0].getDims();
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

    setPostOps(*attr);

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

    mkldnn::memory::desc wgh_candidate(MKLDNNExtensionUtils::convertToDnnlDims(weightsDims), wdt, mkldnn::memory::format_tag::any);

    if (withBiases) {
        mkldnn::memory::desc bias_candidate(MKLDNNExtensionUtils::convertToDnnlDims(inputShapes[BIAS_ID].getStaticDims()), bdt,
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
    createDescriptorInternal(MemoryDescUtils::convertToDnnlMemoryDesc(inputDesc[0])->getDnnlDesc(),
                             MemoryDescUtils::convertToDnnlMemoryDesc(outputDesc[0])->getDnnlDesc());
}

std::shared_ptr<MemoryDesc> MKLDNNFullyConnectedNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? primitive_desc_it.weights_desc(idx - 1) : primitive_desc_it.src_desc(idx);

    if (getInputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(MKLDNNExtensionUtils::DataTypeToIEPrecision(
            static_cast<mkldnn::memory::data_type>(desc.data.data_type)), getInputShapeAtPort(idx));
    }
    return MKLDNNExtensionUtils::makeDescriptor(desc);
}

std::shared_ptr<MemoryDesc> MKLDNNFullyConnectedNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = primitive_desc_it.dst_desc(idx);

    if (getOutputShapeAtPort(idx).getRank() == 3) {
        return std::make_shared<CpuBlockedMemoryDesc>(MKLDNNExtensionUtils::DataTypeToIEPrecision(
            static_cast<mkldnn::memory::data_type>(desc.data.data_type)), getOutputShapeAtPort(idx));
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
