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
#include <cpu_memory_desc_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNFullyConnectedNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
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

    const auto inDims = getParentEdgeAt(0)->getShape().getStaticMklDims();
    const auto outDims = getChildEdgeAt(0)->getShape().getStaticMklDims();

    if (inDims.size() == 3) {
        weightsDims = InferenceEngine::SizeVector({static_cast<size_t>(outDims[2]), static_cast<size_t>(inDims[2])});
    } else {
        weightsDims.push_back(outDims[1]);
        for (int i = 1; i < inDims.size(); i++)
            weightsDims.push_back(inDims[i]);
    }
    biasesDims.push_back(weightsDims[0]);

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getShape())) {
        auto in_candidate = mkldnn::memory::desc(inDims, inputDataType, format);
        auto out_candidate = mkldnn::memory::desc(outDims, outputDataType, mkldnn::memory::format_tag::any);

        createDescriptorInternal(in_candidate, out_candidate);
    }
}

void MKLDNNFullyConnectedNode::createPrimitive() {
    if (prim)
        return;

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();
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
}

void MKLDNNFullyConnectedNode::execute(mkldnn::stream strm) {
    if (prim) {
        auto reshapeMemory = [this](int argType) {
            auto param = primArgs.find(argType);
            if (param != primArgs.end()) {
                auto oldMem = param->second;
                auto dims = oldMem.get_desc().dims();
                if (dims.size() == 3) {
                    MKLDNNDims normalizedDims({static_cast<ptrdiff_t>(dims[0] * dims[1]), static_cast<ptrdiff_t>(dims[2])});
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

    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<MKLDNNFakeQuantizeNode *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops);
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
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

std::shared_ptr<mkldnn::primitive_attr> MKLDNNFullyConnectedNode::initPrimitiveAttr() {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());

    setPostOps(*attr, true);

    return attr;
}

// WA: creation MKLDNNMemoryDesc with format == any is prohibited
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
        auto outDims = out_candidate.dims();
        auto normalizedInDims = {inDims[0] * inDims[1], inDims[2]};
        auto normalizedOutDims = {outDims[0] * outDims[1], outDims[2]};
        in_candidate = mkldnn::memory::desc(normalizedInDims, in_candidate.data_type(),
                                         MKLDNNMemory::GetPlainFormatByRank(normalizedInDims.size()));
        out_candidate = mkldnn::memory::desc(normalizedOutDims, out_candidate.data_type(),
                                             MKLDNNMemory::GetPlainFormatByRank(normalizedOutDims.size()));
    }

    mkldnn::memory::desc wgh_candidate(MKLDNNDims(weightsDims), wdt, mkldnn::memory::format_tag::any);

    if (withBiases) {
        mkldnn::memory::desc bias_candidate(inputShapes[BIAS_ID].getStaticMklDims(), bdt, mkldnn::memory::format_tag::any);
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

void MKLDNNFullyConnectedNode::createDescriptor(const std::vector<const MemoryDesc*> &inputDesc,
                                                const std::vector<const MemoryDesc*> &outputDesc) {
    createDescriptorInternal(MemoryDescUtils::convertToMKLDNNMemoryDesc(*inputDesc[0]), MemoryDescUtils::convertToMKLDNNMemoryDesc(*outputDesc[0]));
}

std::unique_ptr<MKLDNNMemoryDesc> MKLDNNFullyConnectedNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1)) : MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

    if (getParentEdgeAt(idx)->getShape().getRank() == 3) {
        desc = MKLDNNMemoryDesc(getParentEdgeAt(idx)->getShape().getStaticMklDims(), MKLDNNExtensionUtils::IEPrecisionToDataType(desc.getPrecision()),
                                MKLDNNMemory::GetPlainFormatByRank(getParentEdgeAt(idx)->getShape().getRank()));
    }
    return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(std::move(desc));
}

std::unique_ptr<MKLDNNMemoryDesc> MKLDNNFullyConnectedNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    auto desc = MKLDNNMemoryDesc(primitive_desc_it.dst_desc(idx));
    if (getChildEdgeAt(idx)->getShape().getRank() == 3) {
        desc = MKLDNNMemoryDesc(getChildEdgeAt(idx)->getShape().getStaticMklDims(), MKLDNNExtensionUtils::IEPrecisionToDataType(desc.getPrecision()),
                                MKLDNNMemory::GetPlainFormatByRank(getChildEdgeAt(idx)->getShape().getRank()));
    }
    return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(std::move(desc));
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

    return MKLDNNExtensionUtils::getMaxPrecision(inputPrecisions);
}

REG_MKLDNN_PRIM_FOR(MKLDNNFullyConnectedNode, FullyConnected);
