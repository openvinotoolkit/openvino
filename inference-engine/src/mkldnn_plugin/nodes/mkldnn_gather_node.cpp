// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "mkldnn_gather_node.h"
#include <ngraph/opsets/opset1.hpp>
#include <precision_utils.h>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNGatherNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto gatherOp = ngraph::as_type_ptr<const ngraph::op::v1::Gather>(op);
        if (!gatherOp) {
            errorMessage = "Only opset1 Gather operation is supported";
            return false;
        }

        auto axesOp = gatherOp->get_input_node_shared_ptr(GATHER_AXIS);
        if (!ngraph::as_type_ptr<const ngraph::op::Constant>(axesOp)) {
            errorMessage = "Only Constant operation on 'axis' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNode::MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    errorPrefix_ = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto gatherOp = ngraph::as_type_ptr<ngraph::op::v1::Gather>(op);
    if (gatherOp->get_input_size() != 3 || gatherOp->get_output_size() != 1)
        IE_THROW() << errorPrefix_ << "has incorrect number of input/output edges!";

    const SizeVector& dictionary_dims = gatherOp->get_input_shape(GATHER_DICTIONARY);
    if (dictionary_dims.size() == 0)
        IE_THROW() << errorPrefix_ << "has incorrect input parameters dimension!";

    axis = static_cast<int>(gatherOp->get_axis());
    if (axis < 0)
        axis += dictionary_dims.size();
    // Dictionary must be at least rank axis + 1
    if (!(-static_cast<int>(dictionary_dims.size()) <= axis && axis < static_cast<int>(dictionary_dims.size())))
        IE_THROW() << errorPrefix_ << "has incorrect input parameters dimensions and axis number!";

    //  Find number of dictionaries, index range and data length
    for (int i = 0; i < axis; i++)
        numDictionaries *= dictionary_dims[i];
    indexRange = dictionary_dims[axis];
    for (size_t i = axis + 1; i < dictionary_dims.size(); i++)
        dataLength *= dictionary_dims[i];

    if (dataLength == 0)
        IE_THROW() << errorPrefix_ << "had incorrect input parameters dimension!";
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inIdxPrecision = getOriginalInputPrecisionAtPort(GATHER_INDEXES);
    if (inIdxPrecision != Precision::FP32 && inIdxPrecision != Precision::I32 && inIdxPrecision != Precision::FP16)
        inIdxPrecision = Precision::I32;

    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DICTIONARY);

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, dataPrecision},
                          {TensorDescCreatorTypes::ncsp, inIdxPrecision},
                          {TensorDescCreatorTypes::ncsp, Precision::I32}},
                         {{TensorDescCreatorTypes::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

template <typename index_t, class Conversion>
void MKLDNNGatherNode::gather() {
    size_t src_indexSize = getParentEdgeAt(GATHER_INDEXES)->getBlob()->size();
    size_t outputSize = getChildEdgeAt(0)->getBlob()->byteSize();
    const auto *src_index = reinterpret_cast<const index_t *>(getParentEdgeAt(GATHER_INDEXES)->getMemoryPtr()->GetPtr());
    const auto *src_dataDict = reinterpret_cast<const uint8_t *>(getParentEdgeAt(GATHER_DICTIONARY)->getMemoryPtr()->GetPtr());
    auto *dst_data = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    size_t len = dataLength * getParentEdgeAt(GATHER_DICTIONARY)->getDesc().getPrecision().size();

    parallel_for(src_indexSize, [&](size_t i) {
        unsigned int idx = Conversion()(src_index[i]);

        //  Index clipping
        if (idx < indexRange) {
            //  Copying data to destination from Dictionary
            for (size_t j = 0; j < numDictionaries; j++) {
                cpu_memcpy_s(&dst_data[len * (i + j * src_indexSize)],
                            outputSize - (len * (i + j * src_indexSize)),
                            &src_dataDict[len * (idx + j * indexRange)],
                            len);
            }
        } else {
            for (size_t j = 0; j < numDictionaries; j++) {
                memset(&dst_data[len * (i + j * src_indexSize)], 0, len);
            }
        }
    });
}

void MKLDNNGatherNode::execute(mkldnn::stream strm) {
    switch (getParentEdgeAt(GATHER_INDEXES)->getDesc().getPrecision()) {
        case Precision::FP32:
            gather<float, f32toUi32>();
            break;
        case Precision::I32:
            gather<int32_t, i32toUi32>();
            break;
        default:
            return IE_THROW() << "Unsupported indices input precision";
    }
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
