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
#include "ie_parallel.hpp"

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
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

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

    auto same = [&] (mkldnn::memory::data_type inputDT, mkldnn::memory::data_type outputDT, memory::format fmt) -> PrimitiveDescInfo {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = (!i && canBeInPlace()) ? 0 : -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDT, fmt);
            config.inConfs.push_back(dataConfig);
        }

        InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDT, fmt);
            config.outConfs.push_back(dataConfig);
        return {config, impl_desc_type::ref};
    };

    for (const auto& format : getAvailableFormatsForDims(getChildEdgeAt(0)->getDims())) {
        if (getCnnLayer()->precision == Precision::FP32) {
            mkldnn::memory::data_type inputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision::FP32);
            mkldnn::memory::data_type outputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision::FP32);
            supportedPrimitiveDescriptors.push_back(same(inputDT, outputDT, format));
        } else {
            THROW_IE_EXCEPTION << "Invalid Eltwise layer precision";
        }
    }

    if (getCnnLayer()->precision == Precision::I8) {
        mkldnn::memory::data_type inputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision::U8);
        mkldnn::memory::data_type outputDT = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision::U8);
        supportedPrimitiveDescriptors.push_back(same(inputDT, outputDT, mkldnn::memory::format::nhwc));
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
        try {
            auto primitive_desc = sum::primitive_desc(dstMemPtr->GetDescriptor(), sum_scales, srcs_pd);
            prim = std::shared_ptr<sum>(new sum(primitive_desc, srcs_p, dstMemPtr->GetPrimitive()));
        } catch (...) {
            std::cerr << "Handle this problem correctly!" << std::endl;
            prim = nullptr;
        }
    }
}

void MKLDNNEltwiseNode::initOptimalPrimitiveDescriptor() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (isInitConfig(config))
        return;

    MKLDNNNode::initOptimalPrimitiveDescriptor();

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }

    auto& selectedConfig = getSelectedPrimitiveDescriptor()->getConfig();
    for (size_t i = 1; i < selectedConfig.inConfs.size(); i++) {
        if (selectedConfig.inConfs[0].desc.getPrecision() != selectedConfig.inConfs[i].desc.getPrecision()) {
            selectedConfig.inConfs[i].desc.setPrecision(selectedConfig.inConfs[0].desc.getPrecision());
        }
    }
}

template <typename T0, typename T1> void MKLDNNEltwiseNode::ref_eltwise(int in0, int in1) {
    IE_ASSERT(getParentEdges().size() > 1);

    auto& srcMemory0 = getParentEdgeAt(in0)->getMemory();
    auto& srcMemory1 = getParentEdgeAt(in1)->getMemory();
    const T0 *src0_ptr = reinterpret_cast<const T0*>(srcMemory0.GetData()) +
            srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
    const T1 *src1_ptr = reinterpret_cast<const T1*>(srcMemory1.GetData()) +
            srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
    T0 *dst_ptr = reinterpret_cast<T0*>(getChildEdgeAt(0)->getMemory().GetData()) +
            getChildEdgeAt(0)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
    const size_t dst_data_size = srcMemory0.GetSize() / sizeof(T0) / srcMemory0.GetDims()[0] * batchToProcess();

    if (op == EltwiseLayer::Prod) {
#ifdef _WIN32
        for (int i = 0; i < dst_data_size; i++)
            dst_ptr[i] = src0_ptr[i] * src1_ptr[i];
#else
        parallel_for(dst_data_size, [&](int i) {
            dst_ptr[i] = src0_ptr[i] * src1_ptr[i];
        });
#endif

        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1 *>(getParentEdgeAt(j)->getMemory().GetData()) +
                    getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (int i = 0; i < dst_data_size; i++)
                dst_ptr[i] = dst_ptr[i] * src_ptr[i];
#else
            parallel_for(dst_data_size, [&](int i) {
                dst_ptr[i] = dst_ptr[i] * src_ptr[i];
            });
#endif
        }
    } else if (op == EltwiseLayer::Max)  {
#ifdef _WIN32
        for (int i = 0; i < dst_data_size; i++)
            dst_ptr[i] = std::max(src0_ptr[i], (T0)src1_ptr[i]);
#else
        parallel_for(dst_data_size, [&](int i) {
            dst_ptr[i] = std::max(src0_ptr[i], (T0) src1_ptr[i]);
        });
#endif
        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                    getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (int i = 0; i < dst_data_size; i++)
                dst_ptr[i] = std::max(dst_ptr[i], (T0)src_ptr[i]);
#else
            parallel_for(dst_data_size, [&](int i) {
                dst_ptr[i] = std::max(dst_ptr[i], (T0) src_ptr[i]);
            });
#endif
        }
    } else if (op == EltwiseLayer::Sum)  {
#ifdef _WIN32
        for (int i = 0; i < dst_data_size; i++)
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
#else
        parallel_for(dst_data_size, [&](int i) {
            dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
        });
#endif

        for (int j = 2; j < getParentEdges().size(); j++) {
            const T1 *src_ptr = reinterpret_cast<const T1*>(getParentEdgeAt(j)->getMemory().GetData()) +
                    getParentEdgeAt(j)->getMemory().GetDescriptor().data.layout_desc.blocking.offset_padding;
#ifdef _WIN32
            for (int i = 0; i < dst_data_size; i++)
                dst_ptr[i] = dst_ptr[i] + src_ptr[i];
#else
            parallel_for(dst_data_size, [&](int i) {
                dst_ptr[i] = dst_ptr[i] + src_ptr[i];
            });
#endif
        }
    }
}


void MKLDNNEltwiseNode::execute(mkldnn::stream strm) {
    if (prim) {
        MKLDNNNode::execute(strm);
    } else {
        if (getParentEdges().size() > 2) {
            // Only float supported in this case
            for (int i = 0; i < getParentEdges().size(); i++) {
                if (getParentEdgeAt(i)->getDesc().getPrecision() != Precision::FP32) {
                    THROW_IE_EXCEPTION << "If ref eltwise has more than 2 inputs, only FP32 inputs are supported";
                }
            }

            ref_eltwise<float, float>(0, 1);
            return;
        }

        Precision pi0 = getParentEdgeAt(0)->getDesc().getPrecision();
        Precision pi1 = getParentEdgeAt(1)->getDesc().getPrecision();
        Precision po = getChildEdgeAt(0)->getDesc().getPrecision();

        IE_ASSERT(getParentEdges().size() > 1);

        if (po == Precision::FP32 && pi0 == po && pi1 == po) {
            ref_eltwise<float, float>(0, 1);
        } else if (po == Precision::FP32 && pi0 == po && pi1 == Precision::I8) {
            ref_eltwise<float, int8_t>(0, 1);
        } else if (po == Precision::FP32 && pi1 == po && pi0 == Precision::I8) {
            ref_eltwise<float, int8_t>(1, 0);
        } else if (po == Precision::FP32 && pi0 == po && pi1 == Precision::U8) {
            ref_eltwise<float, uint8_t>(0, 1);
        } else if (po == Precision::FP32 && pi1 == po && pi0 == Precision::U8) {
            ref_eltwise<float, uint8_t>(1, 0);
        } else if (po == Precision::I8 && pi0 == po && pi1 == po) {
            ref_eltwise<int8_t, int8_t>(0, 1);
        } else if (po == Precision::I8 && pi0 == po && pi1 == Precision::U8) {
            ref_eltwise<int8_t, uint8_t>(0, 1);
        } else if (po == Precision::I8 && pi1 == po && pi0 == Precision::U8) {
            ref_eltwise<int8_t, uint8_t>(1, 0);
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
