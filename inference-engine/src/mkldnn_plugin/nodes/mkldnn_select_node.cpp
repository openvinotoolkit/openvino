// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include "ie_parallel.hpp"
#include "mkldnn_select_node.h"
#include <nodes/common/tensor_desc_creator.h>
#include <ngraph/opsets/opset1.hpp>
#include <utils/general_utils.h>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNSelectNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto select = std::dynamic_pointer_cast<const ngraph::opset1::Select>(op);
        if (!select) {
            errorMessage = "Only opset1 Select operation is supported";
            return false;
        }
        const auto broadcast = select->get_auto_broadcast();
        if (!MKLDNNPlugin::one_of(broadcast, ngraph::op::AutoBroadcastSpec::NONE, ngraph::op::AutoBroadcastSpec::NUMPY)) {
            errorMessage = "Does not support broadcast type: " + ngraph::as_string(broadcast.m_type);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNSelectNode::MKLDNNSelectNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Select layer with name '" + op->get_friendly_name() + "'";
    const auto select = std::dynamic_pointer_cast<const ngraph::opset1::Select>(op);

    if (op->get_input_size() != numOfInputs || op->get_output_size() != 1)
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";

    const auto broadcast = select->get_auto_broadcast();
    if (broadcast == ngraph::op::AutoBroadcastSpec::NONE) {
        broadcastType = SelectBroadcastType::NONE;
    } else if (broadcast == ngraph::op::AutoBroadcastSpec::NUMPY) {
        broadcastType = SelectBroadcastType::NUMPY;
    } else {
        IE_THROW() << errorPrefix << " has unsupported broadcast type: " + ngraph::as_string(broadcast.m_type);
    }

    auto conditionShapes = op->get_input_shape(CONDITION);
    if (ngraph::is_scalar(conditionShapes))
        conditionShapes = ngraph::Shape{1};
    auto thenShapes = op->get_input_shape(THEN);
    if (ngraph::is_scalar(thenShapes))
        thenShapes = ngraph::Shape{1};
    auto elseShapes = op->get_input_shape(ELSE);
    if (ngraph::is_scalar(elseShapes))
        elseShapes = ngraph::Shape{1};
    auto outputShapes = op->get_output_shape(0);
    if (ngraph::is_scalar(outputShapes))
        outputShapes = ngraph::Shape{1};

    if (broadcastType == SelectBroadcastType::NONE && ((conditionShapes != outputShapes) || (thenShapes != outputShapes) ||
                                                       (elseShapes != outputShapes)))
        IE_THROW() << errorPrefix << " and auto_broadcast='none' has input shapes mismatch";

    if (broadcastType == SelectBroadcastType::NUMPY) {
        if (outputShapes.size() < conditionShapes.size() || outputShapes.size() < thenShapes.size() || outputShapes.size() < elseShapes.size())
            IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible input and output shapes";

        for (int condIt = conditionShapes.size() - 1, outIt = outputShapes.size() - 1; condIt >= 0; condIt--, outIt--)
            if (conditionShapes[condIt] != outputShapes[outIt] && conditionShapes[condIt] != 1)
                IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Condition' input and output shapes";

        for (int thenIt = thenShapes.size() - 1, outIt = outputShapes.size() - 1; thenIt >= 0; thenIt--, outIt--)
            if (thenShapes[thenIt] != outputShapes[outIt] && thenShapes[thenIt] != 1)
                IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Then' input and output shapes";

        for (int elseIt = elseShapes.size() - 1, outIt = outputShapes.size() - 1; elseIt >= 0; elseIt--, outIt--)
            if (elseShapes[elseIt] != outputShapes[outIt] && elseShapes[elseIt] != 1)
                IE_THROW() << errorPrefix << " and auto_broadcast='numpy' has incompatible 'Else' input and output shapes";
    }

    resDims.resize(numOfDims, 1);
    std::copy(std::begin(outputShapes), std::end(outputShapes), std::begin(resDims) + (numOfDims - outputShapes.size()));
    if (broadcastType == SelectBroadcastType::NUMPY) {
        calcOutOffset(resOffset, resDims);

        std::vector<size_t> condDims(numOfDims, 1);
        std::copy(std::begin(conditionShapes), std::end(conditionShapes), std::begin(condDims) + (numOfDims - conditionShapes.size()));
        calcInOffset(condOffset, condDims, resDims);

        std::vector<size_t> thenDims(numOfDims, 1);
        std::copy(std::begin(thenShapes), std::end(thenShapes), std::begin(thenDims) + (numOfDims - thenShapes.size()));
        calcInOffset(thenOffset, thenDims, resDims);

        std::vector<size_t> elseDims(numOfDims, 1);
        std::copy(std::begin(elseShapes), std::end(elseShapes), std::begin(elseDims) + (numOfDims - elseShapes.size()));
        calcInOffset(elseOffset, elseDims, resDims);
    }
}

void MKLDNNSelectNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto inputThenPrecision = getOriginalInputPrecisionAtPort(THEN);
    const auto inputElsePrecision = getOriginalInputPrecisionAtPort(ELSE);
    auto inputPrecision = inputThenPrecision;
    if (inputThenPrecision == Precision::BF16 || inputElsePrecision == Precision::BF16) {
        inputPrecision = Precision::BF16;
    } else if (inputThenPrecision != inputElsePrecision) {
        IE_THROW() << errorPrefix << " has different precisions on 'Then' and 'Else' inputs ";
    }

    const auto conditionPrecision = getOriginalInputPrecisionAtPort(CONDITION);
    if (conditionPrecision != Precision::BOOL && conditionPrecision != Precision::I32  && conditionPrecision != Precision::U8)
        IE_THROW() << errorPrefix << " has unsupported precision: " << conditionPrecision << " on 'Condition' input";

    const auto inputPrecisionSize = inputPrecision.size();
    if (inputPrecisionSize != 1 && inputPrecisionSize != 2 && inputPrecisionSize != 4 && inputPrecisionSize != 8)
        IE_THROW() << errorPrefix << " has unsupported precision: " << inputPrecision << " on 'Then' and 'Else' inputs";

    addSupportedPrimDesc({{TensorDescCreatorTypes::ncsp, conditionPrecision},
                          {TensorDescCreatorTypes::ncsp, inputPrecision},
                          {TensorDescCreatorTypes::ncsp, inputPrecision}},
                         {{TensorDescCreatorTypes::ncsp, inputPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNSelectNode::calcOutOffset(std::vector<size_t>& offset, const std::vector<size_t>& dims) {
    offset.resize(numOfDims);
    int k = 1;
    for (int i = dims.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

void MKLDNNSelectNode::calcInOffset(std::vector<size_t>& offset, const std::vector<size_t>& inDims, const std::vector<size_t>& outDims) {
    offset.resize(numOfDims);
    int k = 1;
    for (int i = inDims.size() - 1; i >= 0; i--) {
        offset[i] = (inDims[i] == outDims[i]) ? k : 0;
        k *= inDims[i];
    }
}

template <typename COND_T, typename DATA_T>
void MKLDNNSelectNode::execute_impl() {
    const auto *conditionData = reinterpret_cast<const COND_T *>(getParentEdgeAt(CONDITION)->getMemoryPtr()->GetPtr());
    const auto *thenData = reinterpret_cast<const DATA_T *>(getParentEdgeAt(THEN)->getMemoryPtr()->GetPtr());
    const auto *elseData = reinterpret_cast<const DATA_T *>(getParentEdgeAt(ELSE)->getMemoryPtr()->GetPtr());
    auto *dstData = reinterpret_cast<DATA_T *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    if (broadcastType == SelectBroadcastType::NONE) {
        size_t dstDataSize = std::accumulate(begin(resDims), end(resDims), 1, std::multiplies<size_t>());
        parallel_for(dstDataSize, [&](size_t i) {
            dstData[i] = conditionData[i] ? thenData[i] : elseData[i];
        });
    } else {
        parallel_for4d(resDims[N], resDims[C], resDims[D], resDims[H], [&](int b, int c, int d, int h) {
            for (int w = 0; w < resDims[W]; w++) {
                size_t indexOut = b * resOffset[N] + c * resOffset[C] + d * resOffset[D] + h * resOffset[H] + w * resOffset[W];
                size_t indexCond = b * condOffset[N] + c * condOffset[C] + d * condOffset[D] + h * condOffset[H] + w * condOffset[W];
                size_t indexThen = b * thenOffset[N] + c * thenOffset[C] + d * thenOffset[D] + h * thenOffset[H] + w * thenOffset[W];
                size_t indexElse = b * elseOffset[N] + c * elseOffset[C] + d * elseOffset[D] + h * elseOffset[H] + w * elseOffset[W];
                dstData[indexOut] = conditionData[indexCond] ? thenData[indexThen] : elseData[indexElse];
            }
        });
    }
}

void MKLDNNSelectNode::execute(mkldnn::stream strm) {
    const size_t condPrecSize = getParentEdgeAt(CONDITION)->getDesc().getPrecision().size();
    const size_t inputsPrecSize = getParentEdgeAt(THEN)->getDesc().getPrecision().size();

    switch (condPrecSize) {
        case 1: {
            switch (inputsPrecSize) {
                case 1: { execute_impl<uint8_t, uint8_t>(); break; }
                case 2: { execute_impl<uint8_t, uint16_t>(); break; }
                case 4: { execute_impl<uint8_t, uint32_t>(); break; }
                case 8: { execute_impl<uint8_t, uint64_t>(); break; }
                default:
                    IE_THROW() << "Select layer doesn't support 'Then' and 'Else' inputs' precision: "
                                   + std::string(getParentEdgeAt(THEN)->getDesc().getPrecision().name());
            }
            break;
        }
        case 4: {
            switch (inputsPrecSize) {
                case 1: { execute_impl<int32_t, uint8_t>(); break; }
                case 2: { execute_impl<int32_t, uint16_t>(); break; }
                case 4: { execute_impl<int32_t, uint32_t>(); break; }
                case 8: { execute_impl<int32_t, uint64_t>(); break; }
                default:
                    IE_THROW() << "Select layer doesn't support 'Then' and 'Else' inputs' precision: "
                                  + std::string(getParentEdgeAt(THEN)->getDesc().getPrecision().name());
            }
            break;
        }
        default: {
                IE_THROW() << "Select layer doesn't support 'Condition' inputs' precision: "
                              + std::string(getParentEdgeAt(CONDITION)->getDesc().getPrecision().name());
        }
    }
}

bool MKLDNNSelectNode::created() const {
    return getType() == Select;
}

REG_MKLDNN_PRIM_FOR(MKLDNNSelectNode, Select)
