// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <string>

#include <ngraph/opsets/opset1.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_range_node.h"
#include <utils/general_utils.h>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNRangeNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!MKLDNNPlugin::one_of(op->get_type_info(), ngraph::op::v0::Range::type_info, ngraph::op::v4::Range::type_info)) {
            errorMessage = "Only opset1 and opset4 Range operation is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(RANGE_START)) == nullptr ||
            std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(RANGE_LIMIT)) == nullptr ||
            std::dynamic_pointer_cast<const ngraph::opset1::Constant>(op->get_input_node_shared_ptr(RANGE_DELTA)) == nullptr) {
            errorMessage = "Only const inputs for Range operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNRangeNode::MKLDNNRangeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Range layer with name '" + op->get_friendly_name() + "'";

    if (getOriginalInputsNumber() != 3 || getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    SizeVector start_dims = op->get_input_shape(RANGE_START);
    if (ngraph::shape_size(start_dims) != 1)
        IE_THROW() << errorPrefix << " has start scalar with more than 1 value";

    SizeVector limit_dims = op->get_input_shape(RANGE_LIMIT);
    if (ngraph::shape_size(limit_dims) != 1)
        IE_THROW() << errorPrefix << " has limit scalar with more than 1 value";

    SizeVector delta_dims = op->get_input_shape(RANGE_DELTA);
    if (ngraph::shape_size(delta_dims) != 1)
        IE_THROW() << errorPrefix << " has delta scalar with more than 1 value";

    SizeVector dst_dims = op->get_output_shape(0);
    if (dst_dims.size() > 1)
        IE_THROW() << errorPrefix << " has unsupported rank for output: " << dst_dims.size();
}

void MKLDNNRangeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<DataConfigurator> inDataConf;
    std::vector<DataConfigurator> outDataConf;

    if (!(getOriginalInputPrecisionAtPort(RANGE_START) == Precision::I32 &&
            getOriginalInputPrecisionAtPort(RANGE_LIMIT) == Precision::I32 &&
            getOriginalInputPrecisionAtPort(RANGE_DELTA) == Precision::I32 &&
            getOriginalOutputPrecisionAtPort(0)     == Precision::I32) &&
        !(getOriginalInputPrecisionAtPort(RANGE_START) == Precision::FP32 &&
            getOriginalInputPrecisionAtPort(RANGE_LIMIT) == Precision::FP32 &&
            getOriginalInputPrecisionAtPort(RANGE_DELTA) == Precision::FP32 &&
            getOriginalOutputPrecisionAtPort(0) == Precision::FP32)) {
        inDataConf.reserve(getOriginalInputsNumber());
        for (int i = 0; i < getOriginalInputsNumber(); ++i)
            inDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::FP32);
        outDataConf.reserve(1);
        outDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::FP32);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    } else {
        inDataConf.reserve(getOriginalInputsNumber());
        for (int i = 0; i < getOriginalInputsNumber(); ++i)
            inDataConf.emplace_back(TensorDescCreatorTypes::ncsp);
        outDataConf.reserve(1);
        outDataConf.emplace_back(TensorDescCreatorTypes::ncsp);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    }
}

void MKLDNNRangeNode::execute(mkldnn::stream strm) {
    StatusCode retcode = OK;
    switch (getParentEdgeAt(0)->getDesc().getPrecision()) {
        case Precision::FP32:
            retcode = rangeKernel<float>();
            break;
        case Precision::I32:
            retcode = rangeKernel<int32_t>();
            break;
        default:
            IE_THROW() << "Incorrect output precision. Only FP32 and I32 are supported!";
    }
    if (retcode == PARAMETER_MISMATCH) {
        std::string errorMsg = "Range indexes exceeds data tensor dimension";
        IE_THROW() << errorMsg;
    }
}

template <typename data_t>
InferenceEngine::StatusCode MKLDNNRangeNode::rangeKernel() noexcept {
    size_t dst_size = (getChildEdgesAtPort(0)[0]->getDims())[0];
    data_t* dst_data = reinterpret_cast<data_t *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());
    data_t start = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_START)->getMemoryPtr()->GetPtr())[0];
    data_t limit = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_LIMIT)->getMemoryPtr()->GetPtr())[0];
    data_t delta = reinterpret_cast<const data_t *>(getParentEdgeAt(RANGE_DELTA)->getMemoryPtr()->GetPtr())[0];
    size_t work_amount_dst = static_cast<size_t>(std::floor(std::abs((limit - start) / delta)));
    if (work_amount_dst != dst_size)
        return PARAMETER_MISMATCH;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t iwork = 0, end = 0;
        splitter(work_amount_dst, nthr, ithr, iwork, end);
        data_t dst_value = start + iwork * delta;

        for (; iwork < end; ++iwork, dst_value += delta) {
            dst_data[iwork] = dst_value;
        }
    });
    return OK;
}

bool MKLDNNRangeNode::created() const {
    return getType() == Range;
}

REG_MKLDNN_PRIM_FOR(MKLDNNRangeNode, Range)
