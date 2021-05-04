// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_input_node.h"
#include "common/cpu_memcpy.h"
#include "mkldnn_extension_utils.h"

#include <string>
#include <tuple>
#include <algorithm>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>
#include <ie_ngraph_utils.hpp>
#include <blob_factory.hpp>
#include "caseless.hpp"
#include "common/cpu_memcpy.h"
#include "common/cpu_convert.h"
#include "utils/cpu_utils.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace details;
using namespace ngraph::op;

MKLDNNInputNode::MKLDNNInputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache), origLayer(op) {
    if (!one_of(op->get_type_info(),
            v0::Parameter::type_info,
            v0::Constant::type_info,
            v0::Result::type_info,
            v3::ReadValue::type_info,
            v6::ReadValue::type_info))
        IE_THROW(NotImplemented) << "CPU Input node doesn't support ngraph operation " << op->get_type_name() << " with name " << op->get_friendly_name();

    constant = ConstantType::NoConst;

    auto constOp = ngraph::as_type_ptr<ngraph::op::Constant>(op);
    if (constOp) {
        constant = ConstantType::Const;

        auto dataPrecision = convertPrecision(op->get_element_type());

        size_t shapeSize = ngraph::shape_size(op->get_shape());
        constexpr size_t byte_size{8};
        if (dataPrecision == Precision::BIN) {
            shapeSize = (shapeSize + (byte_size - 1)) / byte_size;
        }

        TensorDesc td(dataPrecision, {shapeSize}, Layout::C);

        constBlob = make_blob_with_precision(td, const_cast<void*>(constOp->get_data_ptr()));

        MKLDNNDims dims(op->get_shape().empty() ? ngraph::Shape(1, 1) : op->get_shape());

        cloneBlobIfRequired(dims, dataPrecision);
     }
}

void MKLDNNInputNode::cloneBlobIfRequired(const MKLDNNDims& dims, const InferenceEngine::Precision& prec) {
    MKLDNNMemoryDesc memDesc(dims, MKLDNNExtensionUtils::IEPrecisionToDataType(prec));

    auto cloneBlob = [&, this] () {
        MKLDNNMemory memory{ getEngine() };
        memory.Create(memDesc, constBlob->buffer());

        MKLDNNMemoryPtr ptr = MKLDNNMemoryPtr(new MKLDNNMemory(getEngine()));
        ptr->Create(memDesc);
        ptr->SetData(memory);

        return ptr;
    };

    auto isBlobAligned = [&, this] () {
        const void *ptr = constBlob->cbuffer().as<const void*>();
        return prec.size() > 1 ? (reinterpret_cast<size_t>(ptr) % prec.size()) == 0 : true;
    };

    // The presence of subnormals is better to determined at IR read time.
    auto hasSubnormals = [&, this] () {
        if (prec == InferenceEngine::Precision::FP32) {
            uint32_t const *u32data = constBlob->cbuffer().as<const uint32_t*>();
            const size_t size = constBlob->byteSize() / prec.size();
            for (size_t i = 0; i < size; ++i) {
                if (u32data[i] && (u32data[i] & (0xFF << 23)) == 0) {
                    return true;
                }
            }
        }
        return false;
    };

    auto blobKey = [this] () {
        char ptr[32];
        snprintf(ptr, sizeof ptr, "%p", constBlob->cbuffer().as<const void*>());
        return getName()
                + "_" + std::to_string(constBlob->byteSize())
                + "_" + ptr;
    };

    const void *data = constBlob->buffer();
    (void)data;

    if (weightCache) {
        memoryPtr = *weightCache->findOrCreate(blobKey(), cloneBlob);
    } else if (isBlobAligned() && !hasSubnormals()) {
        memoryPtr = MKLDNNMemoryPtr(new MKLDNNMemory(getEngine()));
        memoryPtr->Create(memDesc, constBlob->buffer());
    } else {
        memoryPtr = cloneBlob();
    }
}

MKLDNNInputNode::MKLDNNInputNode(const InferenceEngine::SizeVector &dims, const InferenceEngine::Precision &prc, const std::string &name,
                                 const std::string &type, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(type, name, eng, cache) {
    constant = ConstantType::NoConst;
    if (getType() == Input) {
        outDims.emplace_back(dims);
        addOriginalOutputPrecision(prc);
    }  else if (getType() == Output) {
        inDims.emplace_back(dims);
        addOriginalInputPrecision(prc);
    }
}

void MKLDNNInputNode::withMeanImage() {
    isMeanImage = true;
}

const InferenceEngine::Blob::CPtr MKLDNNInputNode::getConstBlob() const {
    return constBlob;
}

MKLDNNMemoryPtr MKLDNNInputNode::getMemoryPtr() const {
    return memoryPtr;
}

void MKLDNNInputNode::getSupportedDescriptors() {
    if (getType() == Input) {
        if (!getParentEdges().empty())
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
    } else if (getType() == Output) {
        if (getParentEdges().size() != 1)
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (!getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
    }
}

void MKLDNNInputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    LayerConfig config;
    config.dynBatchSupport = true;
    if (getType() == Input || getType() == MemoryInput) {
        precision = getOriginalOutputPrecisionAtPort(0);
        if (precision == Precision::U16 || isMeanImage) {
            precision = Precision::FP32;
        }
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        auto mem_tdesc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
        dataConfig.desc = mem_tdesc;
        config.outConfs.push_back(dataConfig);
        // ReadValue operation expects constant input
        if (!getParentEdges().empty()) {
            DataConfig inConfig;
            inConfig.inPlace = -1;
            inConfig.constant = true;
            inConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
            config.inConfs.push_back(inConfig);
        }
    } else if (getType() == Output) {
        precision = getOriginalInputPrecisionAtPort(0);
        if (precision == Precision::U16) precision = Precision::FP32;
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        auto mem_tdesc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType);
        dataConfig.desc = mem_tdesc;
        config.inConfs.push_back(dataConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNInputNode::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
}

bool MKLDNNInputNode::created() const {
    return getType() == Input || getType() == Output;
}

REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Input);
REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Output);
