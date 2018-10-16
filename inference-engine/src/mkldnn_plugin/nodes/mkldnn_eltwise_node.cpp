// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_eltwise_node.h"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNEltwiseNode::MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng) : MKLDNNNode(layer, eng) {}

bool MKLDNNEltwiseNode::isSum() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());
    return eltwiseLayer->_operation == EltwiseLayer::Sum;
}

bool MKLDNNEltwiseNode::isUnitScales() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());

    if (eltwiseLayer->coeff.empty())
        return true;

    for (auto scale : eltwiseLayer->coeff) {
        if (scale != 1.0f)
            return false;
    }

    return true;
}

void MKLDNNEltwiseNode::getSupportedDescriptors() {
    auto * eltwiseLayer = dynamic_cast<EltwiseLayer*>(getCnnLayer().get());

    if (eltwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert eltwise layer.";
    op = eltwiseLayer->_operation;

    if (getParentEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of input edges.";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges.";

    auto outDims = getParentEdgeAt(0)->getDims();
    for (size_t i = 1; i < getParentEdges().size(); i++) {
        auto oDims = getParentEdgeAt(i)->getDims();
        if (outDims.size() != oDims.size() || outDims.ndims() != oDims.ndims())
            THROW_IE_EXCEPTION << "Dimentions of input layers are not equal for " << eltwiseLayer->name;
    }

    bool with_coeffs = !eltwiseLayer->coeff.empty();
    if (op != EltwiseLayer::Sum && with_coeffs)
        THROW_IE_EXCEPTION << "Only sum operation supports operands coefficients";

    if (with_coeffs && eltwiseLayer->coeff.size() != getParentEdges().size())
        THROW_IE_EXCEPTION << "Number of provided coefficients is not equal to number of operands";

    sum_scales.clear();
    for (int i = 0; i < getParentEdges().size(); i++)
        sum_scales.push_back(with_coeffs ? eltwiseLayer->coeff[i] : 1.0f);
}

void MKLDNNEltwiseNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto same = [&] (memory::format fmt) -> PrimitiveDescInfo {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = (!i && canBeInPlace()) ? 0 : -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, fmt);
            config.inConfs.push_back(dataConfig);
        }

        InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
            config.outConfs.push_back(dataConfig);
        return {config, impl_desc_type::ref};
    };

    for (const auto& format : getAvailableFormatsForDims(getChildEdgeAt(0)->getDims())) {
        supportedPrimitiveDescriptors.push_back(same(format));
    }
}

void MKLDNNEltwiseNode::createPrimitive() {
    if (prim)
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor does not set.";

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<primitive::at> srcs_p;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate.";
        }

        if (op == EltwiseLayer::Sum) {
            srcs_pd.push_back(srcMemPtr->GetPrimitiveDescriptor());
            srcs_p.emplace_back(srcMemPtr->GetPrimitive());
        }
    }
    if (op == EltwiseLayer::Sum) {
        auto primitive_desc = sum::primitive_desc(dstMemPtr->GetDescriptor(), sum_scales, srcs_pd);
        prim = std::shared_ptr<sum>(new sum(primitive_desc, srcs_p, dstMemPtr->GetPrimitive()));
    }
}

void MKLDNNEltwiseNode::execute(mkldnn::stream strm) {
    if (prim) {
        MKLDNNNode::execute(strm);
    } else {
        IE_ASSERT(getParentEdges().size() > 1);

        auto& srcMemory0 = getParentEdgeAt(0)->getMemory();
        auto& srcMemory1 = getParentEdgeAt(1)->getMemory();
        const float *src0_ptr = reinterpret_cast<const float*>(srcMemory0.GetData()) +
                srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
        const float *src1_ptr = reinterpret_cast<const float*>(srcMemory1.GetData()) +
                srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
        float *dst_ptr = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemory().GetData()) +
                getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
        const size_t data_size = srcMemory0.GetSize() / sizeof(float) / srcMemory0.GetDims()[0] * batchToProcess();

        if (op == EltwiseLayer::Prod) {
            #pragma omp parallel for
            for (int i = 0; i < data_size; i++)
                dst_ptr[i] = src0_ptr[i] * src1_ptr[i];

            for (int j = 2; j < getParentEdges().size(); j++) {
                const float *src_ptr = reinterpret_cast<const float *>(getParentEdgeAt(j)->getMemory().GetData()) +
                        getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

                #pragma omp parallel for
                for (int i = 0; i < data_size; i++)
                    dst_ptr[i] = dst_ptr[i] * src_ptr[i];
            }
        } else if (op == EltwiseLayer::Max)  {
            #pragma omp parallel for
            for (int i = 0; i < data_size; i++)
                dst_ptr[i] = std::max(src0_ptr[i], src1_ptr[i]);

            for (int j = 2; j < getParentEdges().size(); j++) {
                const float *src_ptr = reinterpret_cast<const float*>(getParentEdgeAt(j)->getMemory().GetData()) +
                        getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;

                #pragma omp parallel for
                for (int i = 0; i < data_size; i++)
                    dst_ptr[i] = std::max(dst_ptr[i], src_ptr[i]);
            }
        }
    }
}

bool MKLDNNEltwiseNode::created() const {
    return getType() == Eltwise;
}

bool MKLDNNEltwiseNode::canBeInPlace() const {
    size_t inPlaceWithParent = getParentEdges().size();
    for (size_t i = 0; i < inPlaceWithParent; i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (!parentEdge->getParent()->isConstant() &&
                parentEdge->getParent()->getChildEdges().size() == 1) {
            inPlaceWithParent = i;
            break;
        }
    }
    // This is WA for MKLDNN implementation
    if (inPlaceWithParent != 0)
        return false;
    MKLDNNDims dims = getParentEdgeAt(0)->getDims();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        if (getChildEdgeAt(cIdx)->getDims() != dims) {
            return false;
        }
    }

    return true;
}
